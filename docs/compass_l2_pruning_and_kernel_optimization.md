# COMPASS L2 Dynamic Pruning & Pipeline Kernel Optimization

## Overview

This document covers two COMPASS pipeline optimizations implemented in March 2026:

1. **L2 Dynamic Pruning (BLASST-style)**: GPU-side block skipping within the COMPASS Triton kernel
2. **Pipeline Kernel Overhead Reduction**: In-place merge kernel + mask_buffer pre-allocation

---

## 1. L2 Dynamic Pruning

### Architecture: Two-Level Pruning

COMPASS uses a two-level pruning strategy:

| Level | Location | Mechanism | Granularity |
|-------|----------|-----------|-------------|
| **L1** | CPU (`select_blocks`) | Cosine similarity + top-p selection | Sub-block (128 tokens) |
| **L2** | GPU (`_compass_jagged_fwd_kernel`) | BLASST online softmax thresholding | Triton tile (BLOCK_N=64 tokens) |

### L2 Skip Logic (in Triton kernel)

```
For each Q tile (BLOCK_M) × KV tile (BLOCK_N):
  1. Compute QK^T → attention scores
  2. Find local max: m_local = max(scores)
  3. Compare with running global max:
     If (m_local - m_global) < ln(λ):
       → SKIP this block (don't compute softmax, PV, or accumulate)
       → Write mask_buffer[...] = 0
  4. Otherwise: compute full attention and update m_global
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold_ln_lambda` | `float` | `log(λ)`, controls L2 skip aggressiveness |
| `MglobalIn` | `[B, H, S_q]` | Running m_global from previous pipeline piece |
| `MglobalOut` | `[B, H, S_q]` | Updated m_global after this piece |
| `Mask` | `[grid_0, H, max_kv_blocks]` int8 | 1=computed, 0=skipped |

### Compute Density Metric

```
Compute_density = total_L2_computed_pairs / total_full_context_pairs × 100%
```

- **Numerator**: Sum of `mask_buffer == 1` entries across all pipeline pieces, all heads
- **Denominator**: Full context KV blocks (before any pruning) × Q blocks × all heads
- This reflects the combined effect of L1 + L2 pruning

### Key Implementation Details

- `m_global` is tracked **across pipeline pieces** via `torch.maximum(historical_m_global, m_global_out)` (GPU async)
- Density calculation is **deferred** — collected in `deferred_density_data` list during the pipeline loop, computed **after** a single `compute_stream.synchronize()` to avoid breaking pipeline overlap
- With `lambda=1e-10` (ln ≈ -23): L2 effectively never skips, density reflects L1 selection only (~60%)
- With `lambda=0.001` (ln ≈ -6.9): L2 aggressively prunes, density drops to 20-50%

---

## 2. Pipeline Kernel Overhead Reduction

### Problem

Between each COMPASS attention kernel, ~4 small GPU kernels were launched per pipeline piece:

| Source | Kernels/piece | Type |
|--------|--------------|------|
| `merge_attention_outputs` | 2 Triton + 2 alloc | `_merge_lse_kernel` + `_merge_output_kernel` + 2× `empty_like` |
| `torch.maximum(m_global)` | 1 elementwise | `BinaryFunctor<float>` |
| `torch.ones(mask_buffer)` | 1 fill | `FillFunctor<int8>` + `cudaMalloc` |

With 4 pieces × 32 layers × 8 chunks = 1024 groups, the launch overhead was significant.

### Solution A: In-place Merge Kernel

**File**: `nanovllm/ops/compass.py` — `_merge_attention_inplace_kernel` + `merge_attention_inplace()`

Replaces the 2-kernel + 2-allocation `merge_attention_outputs` with a **single** fused Triton kernel that writes results **in-place** to `o1` and `lse1`:

```python
# Before: 4 GPU ops
o_merged = torch.empty_like(o1)       # alloc
lse_merged = torch.empty_like(lse1)   # alloc
_merge_lse_kernel[grid](...)          # kernel 1
_merge_output_kernel[grid](...)       # kernel 2

# After: 1 GPU op, 0 allocs
_merge_attention_inplace_kernel[grid](o1, o2, lse1, lse2, ...)
# o1 and lse1 overwritten in-place
```

**Important**: This is COMPASS-only. Other policies continue using `merge_attention_outputs` from `chunked_attention.py`.

### Solution B: mask_buffer Pre-allocation

**File**: `nanovllm/kvcache/sparse/compass.py` — `alloc_policy_metadata()`

Pre-allocates `self._mask_buffer` once during initialization:

```python
# In alloc_policy_metadata:
self._mask_buffer = torch.empty(
    (max_grid_0, num_heads, max_kv_blocks_per_piece),
    device=device, dtype=torch.int8
)

# In pipeline loop (was torch.ones per piece):
mask_buffer = self._mask_buffer[:grid_0, :num_heads, :max_kv_blocks]
mask_buffer.fill_(1)
```

Eliminates 512× `cudaMalloc` calls — only `fill_(1)` kernel remains.

### Performance Results

| Version | compute_prefill | Total prefill |
|---------|-----------------|---------------|
| v1 (`.item()` in loop) | 9.355s | 13.566s |
| v2 (deferred `.item()`) | 7.598s | 12.147s |
| **v3 (in-place merge + mask reuse)** | **7.762s** | **11.960s** |

nsys kernel stats change:
- `_merge_lse_kernel` + `_merge_output_kernel` → **eliminated** (2 kernels → 0)
- `_merge_attention_inplace_kernel` → **288 calls** (1 kernel replaces 2)
- `FillFunctor<signed char>` → 512 → 508 (same count, no malloc overhead)

---

## Files Modified

| File | Changes |
|------|---------|
| `nanovllm/ops/compass.py` | Added `_merge_attention_inplace_kernel` Triton kernel and `merge_attention_inplace()` wrapper |
| `nanovllm/kvcache/sparse/compass.py` | L2 pruning in pipeline, `_mask_buffer` pre-alloc in metadata, in-place merge in pipeline loop, deferred density tracking |
