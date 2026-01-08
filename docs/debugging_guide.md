# Debugging Guide

This document provides debugging techniques for nano-vLLM, including PyTorch hooks for capturing intermediate tensors.

## PyTorch Hooks for Debugging

### Hook Positions in Qwen3

```
decoder_layer
├── input_layernorm (RMSNorm)
├── self_attn (Qwen3Attention)          ← Hook here for attention I/O after o_proj
│   ├── q_proj → q_norm → RoPE
│   ├── k_proj → k_norm → RoPE
│   ├── v_proj
│   ├── attn (Attention)                ← Hook here for Q/K/V tensors
│   │   └── FlashAttention / SDPA
│   └── o_proj
├── post_attention_layernorm (RMSNorm)
└── mlp (Qwen3MLP)
```

### Hook Types & Data Shapes

| Hook Position | Type | Captured Data |
|---------------|------|---------------|
| `self_attn` | post | `[batch, seq_len, hidden_size]` - after o_proj |
| `self_attn.attn` | pre | Q,K,V: `[seq_len, num_heads, head_dim]` - after RoPE |
| `self_attn.attn` | post | `[seq_len, num_heads, head_dim]` - before o_proj |

### Example: Capture Attention Outputs

```python
storage = {}

def make_hook(layer_id: int, storage: dict):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        # nanovllm shape: [num_tokens, hidden_size] -> add batch dim
        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
        storage[layer_id] = attn_output.detach().clone()
    return hook

# Register hooks
hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(make_hook(layer_idx, storage)))

# Run inference...

# Cleanup
for hook in hooks:
    hook.remove()
```

### Reference Implementation

Key files for comparison testing:

| File | Purpose |
|------|---------|
| `tests/modeling_qwen3.py` | Reference Qwen3 implementation (torch + transformers only) |
| `tests/test_needle_ref.py` | Reference needle test using custom Qwen3 |
| `tests/test_needle.py` | Needle-in-haystack test for nanovllm |

### Common Pitfalls

1. **Shape mismatch**: nanovllm uses `[num_tokens, ...]` while torch uses `[batch, seq_len, ...]`
2. **Hook position**: `self_attn` captures after o_proj, `self_attn.attn` captures before o_proj
3. **Output format**: nanovllm returns tuple `(attn_output, None)`, handle with `output[0]`

---

## Memory Debugging

### Track Peak GPU Memory

```python
import torch

# Reset stats before operation
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Run operation
outputs = llm.generate([prompt], sampling_params)

# Check peak
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak GPU memory: {peak_gb:.2f} GB")
```

### Monitor Memory During Execution

```python
import torch

def memory_snapshot():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Add snapshots at key points in your code
```

---

## Comparing Outputs

### Needle-in-Haystack Test

```bash
# Test with CPU offload
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle.py --enable-offload --input-len 8192

# Test without CPU offload (GPU-only)
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle.py --input-len 8192

# Compare with reference implementation
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle_ref.py --input-len 8192
```

### Tensor Comparison

```python
def compare_tensors(a, b, name, rtol=1e-3, atol=1e-5):
    if a.shape != b.shape:
        print(f"{name}: Shape mismatch {a.shape} vs {b.shape}")
        return False

    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    close = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, close={close}")
    return close
```
