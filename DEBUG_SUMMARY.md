# Chunked Prefill Bug Debug Summary

## Problem
`test_needle.py --enable-offload --input-len 8192` fails with garbage output.

The model generates completely wrong tokens instead of the expected "7492".

## Investigation Progress

### 1. Stream Synchronization Fix (Completed)
- Replaced Triton `store_kvcache` kernel with pure PyTorch operations
- Moved `store_kvcache` to `compute_stream` in chunked prefill mode
- Added sync: `compute_stream.wait_event(offload_done)` after per-layer offload
- Added sync: `default_stream.wait_stream(compute_stream)` before return

### 2. KV Cache Alignment Verification (Completed)
Created alignment tests to compare K/V tensors between torch reference and nanovllm:

**RoPE Alignment:**
- RoPE implementations match perfectly (max_diff=0.002, cosine ~1.0)
- Confirmed RoPE is NOT the cause of the bug

**K/V Cache Alignment (Chunk 0):**
- Cosine similarity: ~1.0 for all layers
- Max diff: 2-7 (grows linearly with position, characteristic of FP16 precision)
- Mean diff: < 0.001
- **Conclusion: K/V cache offload is working correctly**

### 3. Layer Output Divergence Analysis (Completed)
Created per-chunk layer output comparison:

**Chunk 0 (tokens 0-4096):**
- All layers pass with excellent cosine similarity (0.999+)
- Max diff grows in later layers but within acceptable range

**Chunk 1 (tokens 4096-8192):**
- Layers 0-19: OK (cosine ~1.0)
- Layers 20-27: Diverge (cosine 0.83-0.96, max_diff up to 114)
- Divergence correlates with later transformer layers

### 4. Critical Discovery: Single-Chunk Offload Also Fails
**Key finding:** Even with input_len=2048 (single chunk, no chunked attention), the model produces garbage output with CPU offload enabled.

```
# Without offload: PASSES
python tests/test_needle.py --input-len 2048
# Output: "7492" (correct)

# With offload: FAILS
python tests/test_needle.py --enable-offload --input-len 2048
# Output: "The Ble White Th G Lopsiswin..." (garbage)
```

**This proves the bug is NOT in:**
- Chunked attention logic (merge_attention_outputs)
- Multi-chunk KV loading
- Ring buffer pipeline

**The bug IS in:**
- The decode path when CPU offload is enabled
- How prefilled KV is loaded/used during decode

### 5. Decode Path Analysis (In Progress)
The decode path in CPU offload mode:
1. Prefill writes KV to GPU, offloads to CPU
2. Decode loads prefilled KV from CPU via `_decode_ring_buffer_pipeline`
3. Attend to prefilled KV + accumulated decode tokens
4. Merge results

**Observations:**
- `prefilled_blocks` set is empty after decode (should contain block IDs)
- CPU cache has valid data (reasonable mean/std values)
- Decode buffer has zeros (decode tokens not being stored correctly?)

## Current Status

### Working
- Stream synchronization fixes
- K/V cache offload to CPU (verified alignment)
- RoPE implementation
- Chunked prefill attention for first chunk

### Not Working
- Decode with CPU offload (even for single-chunk inputs)
- Multi-chunk attention (divergence in later layers for chunk 1)

## Next Steps
1. Debug why `prefilled_blocks` is empty after decode
2. Check if decode path correctly loads KV from CPU
3. Verify decode buffer is being written correctly
4. Compare decode attention outputs between offload and non-offload modes

## Key Files
- `nanovllm/layers/attention.py` - Main attention implementation with chunked paths
- `nanovllm/kvcache/offload_engine.py` - CPU-GPU transfer engine
- `nanovllm/kvcache/hybrid_manager.py` - KV cache management with `prefilled_blocks`
- `nanovllm/engine/model_runner.py` - Prefill/decode orchestration

## Hypothesis
The decode path fails because:
1. `prefilled_blocks` is not being tracked correctly, causing `get_prefilled_cpu_blocks()` to return empty
2. OR the decode attention is not correctly loading/using the prefilled KV from CPU
3. OR there's a stream synchronization issue specific to decode path
