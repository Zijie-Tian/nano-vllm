# Task Plan: Multi-Model Support for nanovllm

## Goal
扩展 nanovllm 框架以支持多种模型（当前只支持 Qwen3），特别是添加 Llama-3.1-8B-Instruct 支持，并建立可扩展的模型添加范式。

## Current State Analysis

### 硬编码问题位置
- `nanovllm/engine/model_runner.py:35`: 直接实例化 `Qwen3ForCausalLM(hf_config)`
- `nanovllm/engine/model_runner.py:9`: 硬编码导入 `from nanovllm.models.qwen3 import Qwen3ForCausalLM`

### Qwen3 vs Llama 3.1 架构差异

| Feature | Qwen3 | Llama 3.1 |
|---------|-------|-----------|
| Config Class | Qwen3Config | LlamaConfig |
| attention_bias | True (可配置) | False |
| q_norm/k_norm | 有 (when bias=False) | 无 |
| mlp_bias | N/A | False |
| RoPE Scaling | None (目前) | llama3 类型 |
| RoPE theta | 1000000 | 500000 |
| hidden_act | silu | silu |
| tie_word_embeddings | True | False |

### 关键限制
- `rotary_embedding.py:59`: `assert rope_scaling is None` - 不支持 RoPE scaling

---

## Phases

### Phase 1: Create Model Registry Pattern [pending]
**Files to modify:**
- `nanovllm/models/__init__.py` (new)
- `nanovllm/models/registry.py` (new)

**Tasks:**
1. 创建模型注册表机制
2. 定义模型注册装饰器 `@register_model`
3. 实现 `get_model_class(hf_config)` 函数，根据 `architectures` 字段自动选择模型

**Design:**
```python
MODEL_REGISTRY: dict[str, type] = {}

def register_model(*architectures):
    """Decorator to register a model class for given architecture names."""
    def decorator(cls):
        for arch in architectures:
            MODEL_REGISTRY[arch] = cls
        return cls
    return decorator

def get_model_class(hf_config) -> type:
    """Get model class based on HF config architectures."""
    for arch in hf_config.architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(f"Unsupported architecture: {hf_config.architectures}")
```

### Phase 2: Add Llama3 RoPE Scaling Support [pending]
**Files to modify:**
- `nanovllm/layers/rotary_embedding.py`

**Tasks:**
1. 实现 `Llama3RotaryEmbedding` 类，支持 llama3 rope_type
2. 修改 `get_rope()` 函数，根据 rope_scaling 类型选择实现
3. 保持向后兼容（rope_scaling=None 使用原实现）

**Llama3 RoPE Scaling Formula:**
```python
# From transformers:
# low_freq_factor, high_freq_factor, original_max_position_embeddings
# Adjust frequencies based on wavelength thresholds
```

### Phase 3: Implement Llama Model [pending]
**Files to create:**
- `nanovllm/models/llama.py`

**Tasks:**
1. 创建 `LlamaAttention` 类（无 q_norm/k_norm，无 QKV bias）
2. 创建 `LlamaMLP` 类（与 Qwen3MLP 类似，无 bias）
3. 创建 `LlamaDecoderLayer` 类
4. 创建 `LlamaModel` 和 `LlamaForCausalLM` 类
5. 添加 `packed_modules_mapping` 以支持权重加载
6. 使用 `@register_model("LlamaForCausalLM")` 注册

### Phase 4: Modify ModelRunner for Dynamic Loading [pending]
**Files to modify:**
- `nanovllm/engine/model_runner.py`

**Tasks:**
1. 移除硬编码 `from nanovllm.models.qwen3 import Qwen3ForCausalLM`
2. 导入 `from nanovllm.models import get_model_class`
3. 替换 `self.model = Qwen3ForCausalLM(hf_config)` 为:
   ```python
   model_class = get_model_class(hf_config)
   self.model = model_class(hf_config)
   ```

### Phase 5: Register Qwen3 Model [pending]
**Files to modify:**
- `nanovllm/models/qwen3.py`

**Tasks:**
1. 导入 `from nanovllm.models.registry import register_model`
2. 添加 `@register_model("Qwen3ForCausalLM", "Qwen2ForCausalLM")` 装饰器

### Phase 6: Test with Llama-3.1-8B-Instruct [pending]
**Files:**
- `tests/test_needle.py` (existing, use for validation)

**Tasks:**
1. 运行 needle 测试: `python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct`
2. 验证模型加载正确
3. 验证推理输出正确

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |

---

## Success Criteria
- [x] 分析完成：理解当前架构和需要的改动
- [ ] Phase 1: 模型注册表实现
- [ ] Phase 2: Llama3 RoPE scaling 支持
- [ ] Phase 3: Llama 模型实现
- [ ] Phase 4: ModelRunner 动态加载
- [ ] Phase 5: Qwen3 模型注册
- [ ] Phase 6: Llama needle 测试通过

---

## Notes
- 保持现有 Qwen3 功能不变
- 遵循现有代码风格
- 复用现有 layers 组件（Linear, RMSNorm, Embedding 等）
- 只添加必要的代码，不过度工程化
