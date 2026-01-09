# Notes: GPU-only Performance Fix

## Core Insight

Offload mode uses **contiguous layout** for KV cache:

```python
# OffloadEngine CPU cache
k_cache_cpu: [num_layers, num_blocks, block_size, kv_heads, head_dim]

# Store during prefill - contiguous slice assignment
k_cache_cpu[layer_id, block_id, :size].copy_(k[start:end])
```

GPU-only mode uses **PagedAttention blocked layout**:

```python
# GPUOnlyManager cache
kv_cache: [2, num_layers, num_blocks, block_size, kv_heads, head_dim]

# Store during prefill - scatter via index_copy_
k_cache_flat.index_copy_(0, slot_mapping, k_flat)  # SLOW!
```

## Solution

Add contiguous GPU cache to GPUOnlyManager, mirroring offload's design:

```python
# New contiguous GPU cache
contiguous_k_cache: [num_layers, max_seq_len, kv_heads, head_dim]
contiguous_v_cache: [num_layers, max_seq_len, kv_heads, head_dim]

# Store during prefill - direct slice assignment (no scatter!)
contiguous_k_cache[layer_id, :seq_len] = k
```

## Key Files to Modify

1. **`nanovllm/kvcache/gpu_manager.py`**
   - Add `max_seq_len` parameter to `__init__`
   - Add `contiguous_k_cache`, `contiguous_v_cache` allocation in `allocate_cache`
   - Add `contiguous_seq_len` to track current sequence length

2. **`nanovllm/kvcache/__init__.py`**
   - Pass `max_model_len` to GPUOnlyManager in factory function

3. **`nanovllm/engine/model_runner.py`**
   - Add `run_gpu_only_prefill()` - mirrors `run_layerwise_offload_prefill()`
   - Add `run_gpu_only_decode()` - mirrors `run_layerwise_offload_decode()`
   - Add `_should_use_contiguous_gpu_mode()` check
   - Modify `run()` to route to contiguous path

## Comparison with Offload Mode

| Operation | Offload Mode | GPU-only Contiguous |
|-----------|--------------|---------------------|
| Prefill store | `copy_()` to CPU | `=` assignment to GPU |
| Decode load | H2D via ring buffer | Direct GPU slice |
| Memory location | CPU (pinned) | GPU |
| Transfer | D2H / H2D | None |

GPU-only contiguous should be **faster** than offload because no cross-device transfer!

## Memory Layout

```
Offload (CPU):     [num_layers, num_blocks, block_size, kv_heads, head_dim]
GPU Contiguous:    [num_layers, max_seq_len, kv_heads, head_dim]
GPU PagedAttention: [2, num_layers, num_blocks, block_size, kv_heads, head_dim]
```

GPU contiguous is simpler - no block dimension, just flat sequence.
