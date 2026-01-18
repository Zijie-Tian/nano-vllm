# Optimization Guide

This document describes performance optimizations implemented in nano-vLLM, including sgDMA, Triton fused kernels, and N-way pipeline.

---

## Scatter-Gather DMA (sgDMA) - INTEGRATED ✓

### Problem

Strided CPU cache access `k_cache_cpu[:, block_id]` caused slow Device→Pageable transfers at ~1.4 GB/s instead of optimal ~24 GB/s pinned memory bandwidth.

### Solution

Implemented `cudaMemcpy2D` via custom CUDA extension to handle strided layouts natively.

**Integration complete**: 2025-12-25

### Quick Start

```python
from nanovllm.comm import memcpy_2d_async

# Transfer block_id across all layers
spitch = num_blocks * features * dtype_size  # stride between layers
dpitch = features * dtype_size               # contiguous destination
width = features * dtype_size                # bytes per row
height = num_layers                          # number of rows

memcpy_2d_async(gpu_buf, cpu_cache[:, block_id], dpitch, spitch, width, height, "h2d", stream)
```

### Benchmark Performance (Synthetic, 256MB)

| Method | Bandwidth | Speedup |
|--------|-----------|---------|
| **cudaMemcpy2D (sgDMA)** | **24.95 GB/s** | **Baseline** |
| PyTorch strided | 4.25 GB/s | **5.87x slower** |
| PyTorch contiguous | 24.92 GB/s | Same |

### Real-World Performance (A100, Attention Offload)

**Measured from `test_attention_offload.py` profiling**:

| Transfer Type | Count | Bandwidth | Previous | Speedup |
|---------------|-------|-----------|----------|---------|
| **Device→Pinned (D2H)** | 416 | **21.49 GB/s** | 1.40 GB/s | **15.35x** |
| **Pinned→Device (H2D)** | 24,960 | **23.39 GB/s** | N/A | N/A |
| Device→Pageable (D2H) | **0** | N/A | ~40 transfers | **Eliminated** |

**Verification**: All slow Device→Pageable transfers eliminated. System achieves near-optimal PCIe Gen3 x16 bandwidth.

### Files

- `csrc/sgdma_kernel.cu`, `csrc/sgdma.cpp`: CUDA extension
- `nanovllm/comm/sgdma.py`: Python API
- `kvcache/offload_engine.py`: Integration (4 methods updated)

### Build

```bash
python setup.py build_ext --inplace
```

### Integration Details

**Modified methods in `offload_engine.py`**:
- `load_to_slot_all_layers()`: H2D ring buffer load
- `offload_slot_to_cpu()`: D2H ring buffer offload
- `offload_decode_slot()`: D2H decode slot offload
- `load_cpu_blocks_to_gpu_slots_all_layers()`: Batch H2D load

**Example replacement**:
```python
# Before (slow, Device→Pageable fallback)
self.k_cache_gpu[:, slot].copy_(self.k_cache_cpu[:, cpu_block], non_blocking=True)

# After (fast, Device→Pinned via sgDMA)
memcpy_2d_async(
    self.k_cache_gpu[:, slot], self.k_cache_cpu[:, cpu_block],
    self.gpu_pitch, self.cpu_pitch, self.width, self.height,
    "h2d", stream=self.transfer_stream_main
)
```

**Actual Impact**: 15.35x faster D2H transfers, eliminates memory transfer bottleneck. Expected 2-3x overall prefill throughput improvement.

---

## Online Softmax Merge - Triton Fused Kernel ✓

### Problem

Original PyTorch implementation of `merge_attention_outputs()` launches 7 separate kernels per merge operation:

1. `torch.maximum()` - max(lse1, lse2)
2. `torch.exp()` (2x) - exp(lse1-max), exp(lse2-max)
3. `transpose()` + `unsqueeze()` - reshape for broadcasting
4. Accumulation (6x) - weighted sum operations
5. Division - normalize output
6. `torch.log()` - merge LSE
7. `.to()` - type conversion

**Profiling revealed**: In ChunkedPrefill with 8 layers, these operations consumed **698 ms** GPU time (vs FlashAttention 603 ms), becoming a major bottleneck.

### Solution

Implemented Triton fused kernels that combine all operations into 2 kernels.

**Integration complete**: 2025-12-25

### Implementation

**File**: `nanovllm/kvcache/chunked_attention.py:278-408`

Two Triton kernels replace all PyTorch operations:

```python
@triton.jit
def _merge_lse_kernel(...):
    """Fused: max + exp + log"""
    max_lse = tl.maximum(lse1, lse2)
    exp1 = tl.exp(lse1 - max_lse)
    exp2 = tl.exp(lse2 - max_lse)
    lse_merged = max_lse + tl.log(exp1 + exp2)
    tl.store(lse_out_ptr + offsets, lse_merged, mask=mask)

@triton.jit
def _merge_output_kernel(...):
    """Fused: broadcast + weighted sum + division"""
    # Load LSE, compute scaling factors
    exp1 = tl.exp(lse1 - max_lse)
    exp2 = tl.exp(lse2 - max_lse)
    sum_exp = exp1 + exp2

    # Process headdim in chunks
    for d_offset in range(0, headdim, BLOCK_SIZE):
        o1_val = tl.load(o1_ptr + o_idx, mask=mask)
        o2_val = tl.load(o2_ptr + o_idx, mask=mask)
        o_merged = (o1_val * exp1 + o2_val * exp2) / sum_exp
        tl.store(o_out_ptr + o_idx, o_merged, mask=mask)
```

### Performance Results

**From `test_attention_offload.py` profiling** (8 layers, 16K tokens, 16 chunks, 10 iterations):

| Metric | PyTorch (7 kernels) | Triton (2 kernels) | Speedup |
|--------|---------------------|---------------------|---------|
| **GPU time (8 layers)** | 698 ms | 160.7 ms | **4.3x** |
| **Per-layer time** | 87.3 ms | 20.1 ms | **4.3x** |
| **Avg per merge** | 56 µs | 12.9 µs | **4.3x** |
| **Kernel launches** | 10,920 | 3,120 | **71% reduction** |

**Breakdown** (per-layer, 1,560 merges):
- `_merge_output_kernel`: 126.9 ms / 8 = 15.9 ms/layer (avg 10.2 µs/call)
- `_merge_lse_kernel`: 33.8 ms / 8 = 4.2 ms/layer (avg 2.7 µs/call)

### Overall ChunkedPrefill Impact

**GPU time distribution** (test_attention_offload.py):

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| FlashAttention | 603.2 | 74.8% |
| Triton Merge | 160.7 | 19.9% |
| Other | 42.1 | 5.3% |
| **Total** | **806.0** | **100%** |

**If using PyTorch merge** (estimated):
- Total GPU time: ~1,343 ms
- **Overall speedup with Triton**: 1.67x

### Key Files

- `nanovllm/kvcache/chunked_attention.py`: Triton kernels + merge function

---

## N-way Pipeline with Dedicated Streams ✓

### Problem

Original implementation used only 2-slot double buffering, limiting compute-transfer overlap.

### Solution

Implemented N-way pipeline using all available GPU slots with per-slot transfer streams and dedicated compute stream.

**Integration complete**: 2025-12-25

### Architecture

```
Transfer Streams: [slot_0_stream] [slot_1_stream] ... [slot_N_stream]
                       ↓              ↓                    ↓
GPU Slots:          [slot_0]      [slot_1]    ...     [slot_N]
                       ↓              ↓                    ↓
Compute Stream:    ←←←←←←←←←←←← [dedicated compute stream] →→→→→→→→→→→→
```

### Key Design Decisions

1. **Per-slot transfer streams**: Each GPU slot has its own CUDA stream for H2D transfers, enabling parallel loading

2. **Dedicated compute stream**: Created with `torch.cuda.Stream()` (NOT `current_stream()`) to avoid implicit synchronization with CUDA default stream

3. **CUDA Events**:
   - `ring_slot_ready`: Signals transfer complete
   - `ring_slot_compute_done`: Signals safe to overwrite slot

### Performance Impact

**2.0x improvement**: 7.2k → 14.1k tok/s (16K tokens prefill)

---

## Overall Performance Summary

### Completed Optimizations ✓

| Optimization | Date | Impact |
|--------------|------|--------|
| **sgDMA Integration** | 2025-12-25 | 15.35x faster memory transfers (21-23 GB/s) |
| **Triton Fused Merge** | 2025-12-25 | 4.3x faster merges, 1.67x overall ChunkedPrefill |
| **N-way Pipeline** | 2025-12-25 | 2.0x prefill throughput improvement |

### Current Bottlenecks

**From profiling** (`test_attention_offload.py`, 8 layers, 16K tokens):

| Component | GPU Time | Percentage | Optimization Potential |
|-----------|----------|------------|------------------------|
| FlashAttention | 603 ms | 74.8% | ⚠️ Main bottleneck |
| Triton Merge | 161 ms | 19.9% | ✓ Optimized |
| Other | 42 ms | 5.3% | Minor |

### Future Optimization Directions

1. **FlashAttention Optimization** (highest priority)
   - Current: 74.8% of GPU time
   - Potential: Custom FlashAttention kernel for chunked case
   - Expected: 1.5-2x additional speedup

2. **Alternative to sgDMA** (lower priority, PyTorch-only)
   - Reorganize cache layout: `[num_cpu_blocks, num_layers, ...]` instead of `[num_layers, num_cpu_blocks, ...]`
   - Trade-off: Extensive refactoring vs minimal sgDMA approach
   - Same performance as sgDMA (~24 GB/s)

---

**Author**: Zijie Tian
