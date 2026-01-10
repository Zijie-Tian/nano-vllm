# Multi-Model Support

本文档描述 nanovllm 的多模型支持架构，以及如何添加新模型。

## 概述

nanovllm 通过模型注册表 (Model Registry) 机制支持多种模型架构。系统根据 HuggingFace config 中的 `architectures` 字段自动选择对应的模型实现。

### 当前支持的模型

| 架构 | 模型示例 | 文件 |
|------|---------|------|
| `Qwen3ForCausalLM` | Qwen3-0.6B, Qwen3-4B | `nanovllm/models/qwen3.py` |
| `Qwen2ForCausalLM` | Qwen2.5-7B | `nanovllm/models/qwen3.py` |
| `LlamaForCausalLM` | Llama-3.1-8B-Instruct | `nanovllm/models/llama.py` |

## 架构设计

### 模型注册表

```
nanovllm/models/
├── __init__.py      # 导出 get_model_class, 导入所有模型
├── registry.py      # 注册表核心: MODEL_REGISTRY, @register_model
├── qwen3.py         # Qwen3/Qwen2 实现
└── llama.py         # Llama 实现
```

### 动态模型加载流程

```
LLM(model_path)
  → Config.__post_init__()
    → hf_config = AutoConfig.from_pretrained(model_path)
  → ModelRunner.__init__()
    → model_class = get_model_class(hf_config)  # 根据 architectures 选择
    → model = model_class(hf_config)
    → load_model(model, model_path)
```

## 添加新模型

### 步骤 1: 创建模型文件

在 `nanovllm/models/` 下创建新文件，例如 `mistral.py`:

```python
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.registry import register_model


class MistralAttention(nn.Module):
    def __init__(self, ...):
        # 实现注意力层
        pass

class MistralMLP(nn.Module):
    def __init__(self, ...):
        # 实现 MLP 层
        pass

class MistralDecoderLayer(nn.Module):
    def __init__(self, config):
        # 组合 Attention + MLP
        pass

class MistralModel(nn.Module):
    def __init__(self, config):
        # Embedding + Layers + Norm
        pass

@register_model("MistralForCausalLM")
class MistralForCausalLM(nn.Module):
    # 权重映射 (HF 权重名 -> nanovllm 权重名)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = MistralModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

### 步骤 2: 注册模型

在 `nanovllm/models/__init__.py` 中导入新模型:

```python
from nanovllm.models import mistral  # 添加这行
```

### 步骤 3: 处理特殊配置

如果模型有特殊的 RoPE scaling 或其他配置，需要在相应的 layer 中添加支持。

## 模型架构差异

### Qwen3 vs Llama

| 特性 | Qwen3 | Llama |
|------|-------|-------|
| QKV Bias | 可配置 (`attention_bias`) | 无 |
| Q/K Norm | 有 (RMSNorm, 当 bias=False) | 无 |
| MLP Bias | 无 | 无 |
| RoPE Scaling | 无 | llama3 类型 |
| RoPE Theta | 1,000,000 | 500,000 |

### RoPE Scaling 支持

目前支持的 RoPE 类型:

| `rope_type` | 说明 | 模型 |
|-------------|------|------|
| `None` | 标准 RoPE | Qwen3 |
| `llama3` | Llama 3 频率缩放 | Llama 3.1 |

Llama3 RoPE 特点:
- 低频分量 (长距离依赖): 缩放 1/factor
- 高频分量 (短距离依赖): 保持不变
- 中频分量: 平滑插值

## 权重加载

### packed_modules_mapping

nanovllm 将多个 HuggingFace 权重合并到单个张量中以提高效率:

```python
packed_modules_mapping = {
    # HF 权重名: (nanovllm 权重名, shard_id)
    "q_proj": ("qkv_proj", "q"),  # Q 投影 -> QKV 合并
    "k_proj": ("qkv_proj", "k"),  # K 投影 -> QKV 合并
    "v_proj": ("qkv_proj", "v"),  # V 投影 -> QKV 合并
    "gate_proj": ("gate_up_proj", 0),  # Gate -> Gate+Up 合并
    "up_proj": ("gate_up_proj", 1),    # Up -> Gate+Up 合并
}
```

### 权重加载流程

```python
# nanovllm/utils/loader.py
def load_model(model, path):
    for file in glob(path + "/*.safetensors"):
        with safe_open(file) as f:
            for weight_name in f.keys():
                # 检查是否需要映射
                if weight_name in packed_modules_mapping:
                    # 使用自定义 weight_loader
                    param.weight_loader(param, tensor, shard_id)
                else:
                    # 直接复制
                    param.data.copy_(tensor)
```

## 测试验证

### Needle-in-Haystack 测试

```bash
# Llama 3.1 (32K, offload 模式)
CUDA_VISIBLE_DEVICES=0 python tests/test_needle.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --max-model-len 40960 \
    --input-len 32768 \
    --block-size 1024 \
    --num-gpu-blocks 4 \
    --enable-offload

# Qwen3 (8K, offload 模式)
CUDA_VISIBLE_DEVICES=0 python tests/test_needle.py \
    --model ~/models/Qwen3-4B-Instruct-2507 \
    --max-model-len 40960 \
    --input-len 8192 \
    --enable-offload
```

### 测试结果

| 模型 | 输入长度 | Needle 位置 | 结果 |
|------|---------|-------------|------|
| Llama-3.1-8B | 32K | 50% | ✅ PASSED |
| Llama-3.1-8B | 32K | 90% | ✅ PASSED |
| Llama-3.1-8B | 32K | 10% | ❌ FAILED (Lost in Middle) |
| Qwen3-4B | 8K | 50% | ✅ PASSED |

## 文件结构

```
nanovllm/
├── models/
│   ├── __init__.py         # 模型导出和导入
│   ├── registry.py         # 注册表实现
│   ├── qwen3.py           # Qwen3/Qwen2 模型
│   └── llama.py           # Llama 模型
├── layers/
│   ├── rotary_embedding.py # RoPE (含 Llama3 scaling)
│   ├── attention.py        # FlashAttention wrapper
│   ├── linear.py          # 并行 Linear 层
│   └── ...
└── engine/
    └── model_runner.py     # 动态模型加载
```

## 注意事项

1. **Tokenizer 差异**: 不同模型的 tokenizer 分词策略不同，例如 Llama 将 "7492" 分为 2 tokens，Qwen3 分为 4 tokens。

2. **RoPE Scaling**: 如果模型使用非标准 RoPE，需要在 `rotary_embedding.py` 中添加支持。

3. **CPU Offload**: 在 3090 等显存有限的 GPU 上，使用 `--enable-offload` 进行长上下文测试。

4. **Lost in Middle**: LLM 对开头信息的记忆能力较弱，这是模型本身的限制，不是实现问题。
