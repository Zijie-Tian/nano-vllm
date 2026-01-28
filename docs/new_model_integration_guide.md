# 新模型整合指南

本文档总结了将新模型（如GLM-4）整合到nanovllm的经验和常见问题。

## 整合流程概览

```
1. 分析模型配置 (config.json)
   ↓
2. 创建模型文件 (nanovllm/models/<model>.py)
   ↓
3. 实现权重加载 (nanovllm/utils/loader.py)
   ↓
4. 处理特殊组件 (RoPE, Attention, etc.)
   ↓
5. 处理tokenizer差异 (EOS tokens, chat template)
   ↓
6. 验证输出正确性
```

---

## 1. 配置字段映射

不同模型使用不同的配置字段名称，需要建立映射关系：

| 标准字段 | GLM-4 | Qwen | Llama | 说明 |
|----------|-------|------|-------|------|
| `num_key_value_heads` | `multi_query_group_num` | `num_key_value_heads` | `num_key_value_heads` | KV heads数量 |
| `head_dim` | `kv_channels` | 计算得出 | 计算得出 | 每个head的维度 |
| `intermediate_size` | `ffn_hidden_size` | `intermediate_size` | `intermediate_size` | FFN隐藏层大小 |
| `max_position_embeddings` | `seq_length` | `max_position_embeddings` | `max_position_embeddings` | 最大位置 |
| `rope_theta` | `10000 * rope_ratio` | `rope_theta` | `rope_theta` | RoPE基础频率 |

### 代码示例

```python
# 在模型 __init__ 中处理配置差异
num_kv_heads = getattr(config, 'num_key_value_heads',
                       getattr(config, 'multi_query_group_num', num_heads))

head_dim = getattr(config, 'head_dim',
                   getattr(config, 'kv_channels', hidden_size // num_heads))

intermediate_size = getattr(config, 'intermediate_size',
                           getattr(config, 'ffn_hidden_size', None))

max_position = getattr(config, 'max_position_embeddings',
                       getattr(config, 'seq_length', 4096))
```

---

## 2. RoPE实现差异

RoPE是模型整合中**最容易出错**的部分。不同模型可能使用不同的RoPE变体：

### 2.1 旋转方式

| 类型 | 描述 | 使用模型 |
|------|------|----------|
| **Half rotation** | 前半和后半分别旋转 `[x0,x1,...] → [x0*cos-x_{d/2}*sin, ...]` | Llama, Qwen |
| **Interleaved rotation** | 相邻元素配对旋转 `[x0,x1,...] → [x0*cos-x1*sin, x1*cos+x0*sin, ...]` | GLM-4 |

### 2.2 旋转维度

| 类型 | 描述 | 使用模型 |
|------|------|----------|
| **Full rotation** | 旋转整个head_dim | Llama, Qwen |
| **Partial rotation** | 只旋转head_dim的一部分，其余pass-through | GLM-4 (rotary_dim = head_dim // 2) |

### 2.3 GLM-4 RoPE实现

```python
class GLM4RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rotary_dim, ...):
        # GLM-4只旋转一半维度
        self.rotary_dim = rotary_dim  # = head_dim // 2

    def forward(self, positions, query, key):
        # 分离旋转部分和pass-through部分
        q_rot = query[..., :self.rotary_dim]
        q_pass = query[..., self.rotary_dim:]

        # 只对旋转部分应用interleaved RoPE
        q_rot = apply_rotary_emb_interleaved(q_rot, cos, sin)

        # 拼接回去
        return torch.cat([q_rot, q_pass], dim=-1), ...
```

### 2.4 调试RoPE问题

**症状**：模型输出乱码或重复无意义的内容（如 "The. The. The..."）

**调试方法**：
```python
# 对比HuggingFace参考实现的输出
hf_q, hf_k = hf_model.apply_rotary_pos_emb(query, key, cos, sin)
my_q, my_k = my_rotary_emb(positions, query, key)

print(f"Q max diff: {(hf_q - my_q).abs().max()}")  # 应该 < 1e-5
print(f"K max diff: {(hf_k - my_k).abs().max()}")  # 应该 < 1e-5
```

---

## 3. 权重名称映射

不同模型的权重命名规范不同：

### 3.1 常见映射

| 组件 | Llama/Qwen | GLM-4 |
|------|------------|-------|
| Attention QKV | `q_proj`, `k_proj`, `v_proj` | `query_key_value` (合并) |
| Attention Output | `o_proj` | `dense` |
| MLP Gate | `gate_proj` | `dense_h_to_4h` (部分) |
| MLP Up | `up_proj` | `dense_h_to_4h` (部分) |
| MLP Down | `down_proj` | `dense_4h_to_h` |
| LayerNorm | `input_layernorm` | `input_layernorm` |
| Post-Attention LN | `post_attention_layernorm` | `post_attention_layernorm` |

### 3.2 实现权重转换

```python
def convert_glm4_weights(name, param):
    """将GLM-4权重名称转换为nanovllm格式"""
    # 处理合并的QKV权重
    if "query_key_value" in name:
        # 拆分为q, k, v
        q, k, v = param.split([q_size, kv_size, kv_size], dim=0)
        return {"q_proj": q, "k_proj": k, "v_proj": v}

    # 处理合并的gate+up权重
    if "dense_h_to_4h" in name:
        gate, up = param.chunk(2, dim=0)
        return {"gate_proj": gate, "up_proj": up}

    return {name: param}
```

---

## 4. EOS Token处理

### 4.1 问题

某些模型使用**多个EOS tokens**：

| 模型 | EOS Token(s) | 说明 |
|------|--------------|------|
| Llama | `128001` | 单一EOS |
| Qwen | `151643` | 单一EOS |
| GLM-4 | `[151329, 151336, 151338]` | 多个：endoftext, user, observation |

**问题**：`tokenizer.eos_token_id` 只返回第一个，导致模型不会在其他EOS token处停止。

### 4.2 解决方案

```python
# config.py - 支持多个EOS
eos: int | list[int] = -1

# llm_engine.py - 从hf_config读取完整EOS列表
eos_from_config = getattr(config.hf_config, 'eos_token_id', None)
if eos_from_config is not None:
    config.eos = eos_from_config
else:
    config.eos = self.tokenizer.eos_token_id

# scheduler.py - 使用set进行高效查找
self.eos_set = set(eos) if isinstance(eos, list) else {eos}

# 检查时使用 in 而不是 ==
if token_id in self.eos_set:
    # 停止生成
```

### 4.3 调试EOS问题

**症状**：模型总是生成到max_tokens才停止

**调试方法**：
```python
# 检查EOS配置
print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
print(f"hf_config.eos_token_id: {config.hf_config.eos_token_id}")

# 检查输出中的EOS tokens
output = llm.generate([prompt], params)
for eos_id in [151329, 151336, 151338]:
    if eos_id in output[0]['token_ids']:
        print(f"Found EOS {eos_id} at position {output[0]['token_ids'].index(eos_id)}")
```

---

## 5. Chat Template

不同模型使用不同的对话模板：

| 模型 | 模板格式 |
|------|----------|
| Llama-3 | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\n{content}<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n` |
| Qwen | `<\|im_start\|>user\n{content}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| GLM-4 | `[gMASK]<sop><\|user\|>\n{content}<\|assistant\|>\n` |

### 实现模板转换

```python
def convert_to_model_prompt(prompt: str, model_type: str) -> str:
    """将标准prompt转换为模型特定格式"""
    if model_type == "glm4":
        return f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n"
    elif model_type == "llama3":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    # ...
```

---

## 6. 验证清单

整合新模型后，按以下顺序验证：

### 6.1 权重加载验证

```python
# 检查所有权重是否正确加载
for name, param in model.named_parameters():
    if param.abs().sum() == 0:
        print(f"WARNING: {name} is all zeros!")
```

### 6.2 单层输出验证

```python
# 对比embedding层输出
my_emb = my_model.embed_tokens(input_ids)
hf_emb = hf_model.model.embed_tokens(input_ids)
print(f"Embedding diff: {(my_emb - hf_emb).abs().max()}")  # < 1e-5

# 对比第一层输出
my_out = my_model.layers[0](my_emb, ...)
hf_out = hf_model.model.layers[0](hf_emb, ...)
print(f"Layer 0 diff: {(my_out - hf_out).abs().max()}")  # < 1e-4
```

### 6.3 生成质量验证

```python
# 简单问答测试
prompt = "Hello, how are you?"
output = llm.generate([prompt], SamplingParams(max_tokens=50))
print(output[0]['text'])  # 应该是连贯的回答

# 检查是否正确停止
print(f"Generated {len(output[0]['token_ids'])} tokens (max=50)")
```

### 6.4 RULER基准测试

```bash
# 运行1个sample快速验证
python tests/test_ruler.py --model <path> --num-samples 1

# 验证通过后运行完整测试
python tests/test_ruler.py --model <path> --num-samples 100
```

---

## 7. 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 输出乱码/重复 | RoPE实现错误 | 检查旋转方式(interleaved vs half)和旋转维度(full vs partial) |
| 数值爆炸(NaN/Inf) | 权重加载错误或dtype不匹配 | 检查权重映射，确保dtype一致 |
| 不停止生成 | EOS token处理错误 | 从hf_config读取完整EOS列表 |
| 输出质量差 | LayerNorm或bias缺失 | 检查add_qkv_bias等配置 |
| 位置编码错误 | max_position_embeddings读取错误 | 检查配置字段名称(seq_length等) |

---

## 8. 文件结构

新模型整合需要修改/创建的文件：

```
nanovllm/
├── models/
│   └── <model>.py          # 新建：模型定义
├── layers/
│   └── rotary_embedding.py # 修改：如需特殊RoPE
├── utils/
│   └── loader.py           # 修改：权重加载
├── config.py               # 可能修改：新配置字段
└── engine/
    ├── llm_engine.py       # 可能修改：EOS处理
    └── scheduler.py        # 可能修改：EOS检查
tests/
└── test_ruler.py           # 修改：chat template
```

---

## 附录：GLM-4整合案例

### 遇到的问题及解决

1. **配置字段差异** → 添加getattr fallback链
2. **Interleaved RoPE** → 实现`apply_rotary_emb_interleaved`
3. **Partial rotation (head_dim//2)** → 实现`GLM4RotaryEmbedding`
4. **多EOS tokens** → 修改config/llm_engine/scheduler支持list
5. **合并的QKV权重** → 在loader中拆分

### 关键代码位置

- RoPE实现: `nanovllm/layers/rotary_embedding.py:GLM4RotaryEmbedding`
- 模型定义: `nanovllm/models/glm4.py`
- 权重加载: `nanovllm/utils/loader.py:load_glm4_weights`
- EOS处理: `nanovllm/engine/scheduler.py:eos_set`
