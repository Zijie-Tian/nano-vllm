# Findings: Multi-Model Support Analysis

## Current Architecture Analysis

### Model Loading Flow
```
LLM(model_path)
  → LLMEngine.__init__()
    → Config.__post_init__()
      → hf_config = AutoConfig.from_pretrained(model)
    → ModelRunner.__init__()
      → model = Qwen3ForCausalLM(hf_config)  ← HARDCODED
      → load_model(model, config.model)
```

### Key Files
| File | Purpose |
|------|---------|
| `nanovllm/engine/model_runner.py` | 模型加载和运行 |
| `nanovllm/models/qwen3.py` | Qwen3 模型定义 |
| `nanovllm/utils/loader.py` | safetensors 权重加载 |
| `nanovllm/layers/rotary_embedding.py` | RoPE 实现 |

---

## Llama 3.1 Config Analysis

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "attention_bias": false,
  "mlp_bias": false,
  "head_dim": 128,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "max_position_embeddings": 131072,
  "tie_word_embeddings": false,
  "vocab_size": 128256
}
```

### Llama 3 RoPE Scaling
Llama 3 使用特殊的 RoPE scaling 策略 (`rope_type: "llama3"`)：
- 低频分量保持不变（对应短距离依赖）
- 高频分量线性插值（对应长距离依赖）
- 参数: `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`

参考实现 (transformers):
```python
def _compute_llama3_parameters(config, device, inv_freq):
    factor = config.factor
    low_freq_factor = config.low_freq_factor
    high_freq_factor = config.high_freq_factor
    old_context_len = config.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / factor,
        inv_freq
    )
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama + smooth_factor * inv_freq
    is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama
```

---

## Weight Mapping Analysis

### Qwen3 packed_modules_mapping
```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

### Llama Weight Names (from safetensors)
预期 Llama 权重命名与 Qwen3 类似：
- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`

**结论**: Llama 的 `packed_modules_mapping` 与 Qwen3 相同，可以复用。

---

## Shared Components (Can Reuse)

| Component | File | Notes |
|-----------|------|-------|
| `RMSNorm` | `layers/layernorm.py` | 通用 |
| `SiluAndMul` | `layers/activation.py` | 通用 |
| `Attention` | `layers/attention.py` | FlashAttention wrapper |
| `QKVParallelLinear` | `layers/linear.py` | 支持 bias=False |
| `RowParallelLinear` | `layers/linear.py` | 通用 |
| `MergedColumnParallelLinear` | `layers/linear.py` | 通用 |
| `VocabParallelEmbedding` | `layers/embed_head.py` | 通用 |
| `ParallelLMHead` | `layers/embed_head.py` | 通用 |
| `load_model` | `utils/loader.py` | 通用 |

---

## Llama vs Qwen3 Implementation Diff

### Attention
| Feature | Qwen3Attention | LlamaAttention |
|---------|----------------|----------------|
| QKV bias | 可配置 (attention_bias) | 始终 False |
| q_norm | 有 (when bias=False) | 无 |
| k_norm | 有 (when bias=False) | 无 |
| RoPE | Standard | Llama3 scaled |

### MLP
| Feature | Qwen3MLP | LlamaMLP |
|---------|----------|----------|
| gate/up bias | False | False |
| down bias | False | False |
| hidden_act | silu | silu |

**结论**: Llama MLP 与 Qwen3 MLP 几乎相同，可以直接复用或简化。

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| RoPE 实现错误 | 高 - 导致错误输出 | 参考 transformers 实现，单元测试 |
| 权重映射错误 | 高 - 模型无法加载 | 检查 safetensors 键名 |
| 注册表循环导入 | 中 - 启动失败 | 延迟导入 |
