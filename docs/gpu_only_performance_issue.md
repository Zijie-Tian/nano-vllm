# GPU-only Performance Issue: PagedAttention Scatter Overhead

## Problem Summary

GPU-only mode with MInference is **slower** than CPU offload mode for long-context single-sequence inference:

| Mode | Prefill Speed (32K tokens, Qwen3-4B) |
|------|--------------------------------------|
| GPU-only + MInference | 3383 tok/s |
| Offload + MInference | 5373 tok/s |

This counterintuitive result is caused by **unnecessary `store_kvcache` overhead** in the GPU-only path.

## Root Cause Analysis

### GPU-only Execution Path

```python
# attention.py line 86-110
def forward(self, q, k, v):
    # ALWAYS store to cache first - OVERHEAD HERE
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # ← Always executed

    if context.is_prefill:
        if context.sparse_prefill_policy is not None:
            # MInference: uses k, v directly, NOT k_cache!
            o = sparse_prefill_attention(q, k, v, layer_id)
        else:
            # Full attention: also uses k, v directly
            o = flash_attn_varlen_func(q, k, v, ...)
```

**Key observation**: Prefill attention **never reads from cache** - it uses the computed k, v directly. But `store_kvcache` is always called before attention.

### The `store_kvcache` Overhead

```python
# attention.py line 8-59
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    # 1. Filter invalid slots (conditional logic)
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    valid_keys = key[valid_mask]

    # 2. Reshape for scatter operation
    k_cache_flat = k_cache.view(total_slots, D)
    valid_keys_flat = valid_keys.reshape(-1, D)

    # 3. Scatter write via index_copy_ - EXPENSIVE!
    k_cache_flat.index_copy_(0, valid_slots.long(), valid_keys_flat)
    v_cache_flat.index_copy_(0, valid_slots.long(), valid_values_flat)
```

This scatter operation is called for **every layer** (28 layers for Qwen3-4B), writing **all tokens** (32K) to GPU cache.

### Offload Path (No Such Overhead)

```python
# model_runner.py - run_layerwise_offload_prefill
for layer_id in range(num_layers):
    # QKV projection + RoPE
    q, k = layer.self_attn.rotary_emb(positions, q, k)

    # Sparse attention - directly uses k, v
    attn_output = sparse_prefill_attention(q, k, v, layer_id)

    # Contiguous copy to CPU - no scatter!
    offload_engine.offload_layer_kv_sync(layer_id, k, v, cpu_block_ids, total_tokens)
```

## Memory Layout Comparison

| Aspect | GPU-only (PagedAttention) | Offload (Contiguous) |
|--------|---------------------------|----------------------|
| **Layout** | `[num_blocks, block_size, heads, dim]` | `[seq_len, heads, dim]` |
| **Write pattern** | Scatter via `index_copy_` | Contiguous `copy_()` |
| **Indirection** | slot_mapping lookup | None |
| **Memory efficiency** | High (shared block pool) | Low (reserved per seq) |
| **Write performance** | Slow (memory-bound scatter) | Fast (simple DMA) |

### Why PagedAttention Uses Scatter

PagedAttention is designed for:
1. **Multi-sequence batching**: Different sequences share a block pool
2. **Dynamic memory management**: No need to reserve max_len per sequence
3. **Prefix caching**: Shared KV blocks across sequences

But for **single-sequence long-context** inference, these benefits don't apply, and we only pay the scatter overhead.

## Why `store_kvcache` is Still Needed

Even though prefill attention doesn't read from cache, **decode** does:

```python
# attention.py line 111-114
else:  # decode
    # Reads from cache!
    o = flash_attn_with_kvcache(q, k_cache, v_cache, block_table=...)
```

So `store_kvcache` during prefill is preparing KV cache for future decode steps.

## Potential Optimizations

### Option 1: Async Store After Attention (Low Effort)

Move `store_kvcache` after attention computation and make it async:

```python
def forward(self, q, k, v):
    if context.is_prefill:
        # Compute attention first
        if context.sparse_prefill_policy is not None:
            o = sparse_prefill_attention(q, k, v, layer_id)
        else:
            o = flash_attn_varlen_func(q, k, v, ...)

        # Then store async (overlaps with next layer's QKV)
        if k_cache.numel():
            store_kvcache_async(k, v, k_cache, v_cache, slot_mapping)
    ...
```

**Expected benefit**: Overlap store with compute, ~20-30% improvement.

### Option 2: Contiguous Layout for Single-Sequence Mode (Medium Effort)

Add a "contiguous mode" for single-sequence long-context:

```python
class ContiguousKVCache:
    """Simple contiguous KV cache for single-sequence mode."""
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, dtype):
        self.k_cache = torch.zeros(num_layers, max_seq_len, num_kv_heads, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, max_seq_len, num_kv_heads, head_dim, dtype=dtype)

    def store(self, layer_id, k, v, start_pos):
        # Simple contiguous write - no scatter!
        seq_len = k.shape[0]
        self.k_cache[layer_id, start_pos:start_pos+seq_len] = k
        self.v_cache[layer_id, start_pos:start_pos+seq_len] = v
```

**Expected benefit**: Match or exceed offload performance (~60% improvement).

### Option 3: Fused Store-Attention Kernel (High Effort)

Create a fused Triton kernel that:
1. Computes QKV projection
2. Stores K, V to cache
3. Computes attention

This eliminates memory roundtrips entirely.

**Expected benefit**: Best possible performance, but high implementation complexity.

## Recommended Action

For **single-sequence long-context** workloads (the primary use case for MInference):

1. **Short term**: Use offload mode - it's actually faster!
2. **Medium term**: Implement Option 1 (async store) for quick win
3. **Long term**: Consider Option 2 (contiguous layout) for GPU-only mode

## Performance Measurement

To reproduce the benchmark:

```bash
# GPU-only + MInference
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle.py \
    --model ~/models/Qwen3-4B-Instruct-2507/ \
    --input-len 32768 \
    --enable-minference

# Offload + MInference
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle.py \
    --model ~/models/Qwen3-4B-Instruct-2507/ \
    --input-len 32768 \
    --enable-offload \
    --enable-minference
```

## Related Files

- `nanovllm/layers/attention.py`: `store_kvcache()` and `Attention.forward()`
- `nanovllm/engine/model_runner.py`: `run_layerwise_offload_prefill()`
- `nanovllm/kvcache/offload_engine.py`: `offload_layer_kv_sync()`

## References

- [PagedAttention Paper](https://arxiv.org/abs/2309.06180) - vLLM's memory management
- [MInference Paper](https://arxiv.org/abs/2407.02490) - Sparse prefill attention
