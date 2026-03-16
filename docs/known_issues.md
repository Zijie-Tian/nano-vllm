# Known Issues and Fixes

This document documents bugs that were discovered and fixed in nano-vLLM.

---

## Partial Last Block Bug (FIXED ✓)

### Problem

When prefill token count is not an exact multiple of `block_size`, decode outputs garbage.

### Root Cause

`_chunked_decode_attention` calculated `last_block_valid_tokens` using `len(seq) - 1`, which increases during decode. But CPU blocks are fixed after prefill!

```python
# BUG: len(seq) increases each decode step
total_prefill_tokens = len(seq) - 1  # Wrong!
last_block_valid_tokens = total_prefill_tokens % block_size  # Reads garbage from CPU
```

### Fix

Cache original prefill length in `HybridKVCacheManager.get_prefill_len()`:

```python
# CORRECT: Use cached prefill length
total_prefill_tokens = kvcache_manager.get_prefill_len(seq)  # Fixed value
```

### Files Modified

- `nanovllm/kvcache/hybrid_manager.py`: Added `_prefill_len` dict and `get_prefill_len()` method
- `nanovllm/layers/attention.py`: Use `get_prefill_len()` instead of `len(seq) - 1`

### Verification

Tested with various prefill lengths (not multiples of block_size):
- 100 tokens (block_size=1024)
- 5000 tokens (block_size=4096)
- 15000 tokens (block_size=4096)

All tests now produce correct output.

---

## Block Size 4096 Race Condition (FIXED ✓)

### Problem

`block_size=4096` with multiple chunks produced `index_copy_(): index out of bounds` CUDA error during Chunk 2 processing.

### Root Cause

Race condition between default stream and compute stream. In `_prepare_chunked_offload_chunk()`, `slot_mapping` tensor was created with `non_blocking=True` H2D transfer on the default stream. However, `store_kvcache` runs on `compute_stream`. Without synchronization, `compute_stream` could use `slot_mapping` before its transfer completed, causing corrupted indices.

### Fix

Added explicit stream synchronization in `attention.py`:

```python
if is_chunked_offload:
    compute_stream = context.kvcache_manager.offload_engine.compute_stream
    if k_cache.numel() and v_cache.numel():
        # CRITICAL: Wait for default stream to ensure slot_mapping tensor transfer is complete
        compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(compute_stream):
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

### Verification

Tested block sizes: 512, 1024, 4096, 8192 - all pass.

### Files Modified

- `nanovllm/layers/attention.py`: Added `compute_stream.wait_stream(torch.cuda.default_stream())`

---

## COMPASS Accuracy Degradation with `num-gpu-blocks > 4` (KNOWN, UNFIXED)

### Problem

When `--num-gpu-blocks` is set to 8 or 16, COMPASS produces 0% accuracy on niah_single_1 (32K context). Default value of 4 works correctly (100%).

### Test Results

| `--num-gpu-blocks` | Ring Slots | Chunks (32K) | `cos matmul` time | Accuracy |
|---|---|---|---|---|
| **4** (default) | 4 | 8 | 0.11s (8%) | **100%** ✅ |
| **8** | 8 | 4 | 3.45s (72%) | **0%** ❌ |
| **16** | 16 | 2 | 3.45s (71%) | **0%** ❌ |

### Analysis

More GPU blocks → larger chunk capacity → fewer chunks → each `select_blocks` call processes far more historical KV. The `cos matmul` (Q·K similarity estimation) time increases ~31×, suggesting the CPU scoring becomes unreliable at this scale. Likely causes:

1. **Chunk granularity**: With only 2-4 chunks, `top_p` selection operates on much larger KV history per call, potentially losing fine-grained locality needed for needle-in-haystack.
2. **Pooling quality**: Q/K pooling may produce less representative scores over larger token ranges.

### Workaround

Keep `--num-gpu-blocks` at default (4). The N-stage pipeline code dynamically uses all available slots, so performance benefits scale automatically when this is fixed.

### Files Related

- `nanovllm/kvcache/sparse/compass.py`: `select_blocks()`, `compute_chunked_prefill()`
- `nanovllm/config.py`: `num_gpu_blocks` default

---

## Reporting New Issues

If you discover a new bug, please document it here with:

1. **Problem**: Clear description of the issue
2. **Root Cause**: Analysis of why it happens
3. **Fix**: Code changes to resolve it
4. **Files Modified**: List of affected files
5. **Verification**: How the fix was tested

---

**Author**: Zijie Tian
