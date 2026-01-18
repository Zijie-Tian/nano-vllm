# Debugging Guide

This document covers debugging techniques for nano-vLLM, including PyTorch hooks and common pitfalls.

## PyTorch Hooks for Debugging

### Hook Positions in Qwen3

Understanding where to place hooks is critical for capturing the right data:

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

### Reference Implementation Files

| File | Purpose |
|------|---------|
| `tests/modeling_qwen3.py` | Reference Qwen3 implementation (torch + transformers only) |
| `tests/test_needle_ref.py` | Reference needle test using custom Qwen3 |
| `tests/test_needle.py` | Needle-in-haystack test for nanovllm |

## Common Pitfalls

### 1. Shape Mismatch

**Issue**: nanovllm uses `[num_tokens, ...]` while torch uses `[batch, seq_len, ...]`

**Solution**: Always add/remove batch dimension when comparing:
```python
if tensor.dim() == 2:
    tensor = tensor.unsqueeze(0)  # Add batch dim
```

### 2. Hook Position

**Issue**: `self_attn` captures after o_proj, `self_attn.attn` captures before o_proj

**Solution**: Choose the right hook based on what you need:
- Use `self_attn` for final attention output
- Use `self_attn.attn` for raw Q/K/V tensors

### 3. Output Format

**Issue**: nanovllm returns tuple `(attn_output, None)`

**Solution**: Always access first element:
```python
if isinstance(output, tuple):
    actual_output = output[0]
```

## Tensor Comparison

When comparing tensors between nanovllm and reference implementations:

```python
def compare_tensors(name: str, actual, expected, rtol=1e-3, atol=1e-5):
    """Compare two tensors with reasonable tolerances."""
    if actual.shape != expected.shape:
        print(f"{name}: Shape mismatch - {actual.shape} vs {expected.shape}")
        return False

    max_diff = (actual - expected).abs().max().item()
    mean_diff = (actual - expected).abs().mean().item()
    matches = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    print(f"{name}: {'PASS' if matches else 'FAIL'} (max={max_diff:.6f}, mean={mean_diff:.6f})")
    return matches
```

## Memory Profiling

Track GPU memory usage during inference:

```python
import torch

def get_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    return allocated, reserved

# Before inference
alloc_before, reserved_before = get_gpu_memory()

# Run inference...

# After inference
alloc_after, reserved_after = get_gpu_memory()
print(f"GPU Memory: {alloc_after:.2f} GB allocated, {reserved_after:.2f} GB reserved")
print(f"Peak: {(alloc_after - alloc_before):.2f} GB")
```

---

**Author**: Zijie Tian
