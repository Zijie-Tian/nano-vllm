# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3 models with CPU offload for long-context inference.

## Sparse Attention

For sparse attention related content (block sparse attention, MInference, FlexPrefill, XAttention, AvgPool, etc.), refer to [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md).

## Architecture

### Core Components

- **LLMEngine** (`llm_engine.py`): Main entry, runs prefill-decode loop
- **ModelRunner** (`model_runner.py`): Loads weights, allocates KV cache, CUDA graphs
- **Scheduler** (`scheduler.py`): Two-phase scheduling (prefill → decode)
- **BlockManager** (`block_manager.py`): Paged attention with prefix caching (xxhash), default block size 4096
- **Attention** (`layers/attention.py`): FlashAttention with chunked methods for CPU offload

## CPU Offload System

### Ring Buffer Design

```
GPU Slots: [0]  [1]  [2]  [3]  ...  (unified ring buffer)
Prefill: slot = chunk_idx % N
Decode:  slot[0] = decode, slots[1:] = load previous chunks
```

**Key Files**: `kvcache/offload_engine.py`, `kvcache/hybrid_manager.py`

**Memory Layout**:
- GPU: `[num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]`
- CPU: `[num_layers, num_cpu_blocks, ...]` (pinned memory)

**Key Methods**:
- `load_to_slot_layer(slot, layer, cpu_block)`: Async H2D load
- `offload_slot_to_cpu(slot, cpu_block)`: Async D2H offload
- Per-slot per-layer CUDA events for fine-grained synchronization

**Pipeline**: N-way pipeline with dedicated streams for full compute-transfer overlap. Pipeline depth = N-1 (prefill), (N-1)/2 (decode).

### Stream Architecture

```
Transfer Streams: [slot_0_stream] [slot_1_stream] ... [slot_N_stream]
                       ↓              ↓                    ↓
GPU Slots:          [slot_0]      [slot_1]    ...     [slot_N]
                       ↓              ↓                    ↓
Compute Stream:    ←←←←←←←←←←←← [dedicated compute stream] →→→→→→→→→→→→
```

**Key Design Decisions**:
- **Per-slot transfer streams**: Each GPU slot has its own CUDA stream for H2D transfers, enabling parallel loading
- **Dedicated compute stream**: Created with `torch.cuda.Stream()` (NOT `current_stream()`) to avoid implicit synchronization with default stream
- **CUDA Events**: `ring_slot_ready` (transfer complete), `ring_slot_compute_done` (safe to overwrite)

## Scatter-Gather DMA (sgDMA) - INTEGRATED ✓

### Problem & Solution

**Problem**: Strided CPU cache access `k_cache_cpu[:, block_id]` caused slow Device→Pageable transfers at ~1.4 GB/s instead of optimal ~24 GB/s pinned memory bandwidth.

**Solution**: Implemented `cudaMemcpy2D` via custom CUDA extension to handle strided layouts natively. **Integration complete** as of 2025-12-25.

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

**Build**: `python setup.py build_ext --inplace`

**Files**:
- `csrc/sgdma_kernel.cu`, `csrc/sgdma.cpp`: CUDA extension
- `nanovllm/comm/sgdma.py`: Python API
- `tests/test_sgdma.py`: Standalone benchmark
- `kvcache/offload_engine.py`: Integration (4 methods updated)

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

## Online Softmax Merge - Triton Fused Kernel ✓

### Problem & Solution

**Problem**: Original PyTorch implementation of `merge_attention_outputs()` launches 7 separate kernels per merge operation:
1. `torch.maximum()` - max(lse1, lse2)
2. `torch.exp()` (2x) - exp(lse1-max), exp(lse2-max)
3. `transpose()` + `unsqueeze()` - reshape for broadcasting
4. Accumulation (6x) - weighted sum operations
5. Division - normalize output
6. `torch.log()` - merge LSE
7. `.to()` - type conversion

**Profiling revealed**: In ChunkedPrefill with 8 layers, these operations consumed **698 ms** GPU time (vs FlashAttention 603 ms), becoming a major bottleneck.

**Solution**: Implemented Triton fused kernels that combine all operations into 2 kernels. **Integration complete** as of 2025-12-25.

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

### Correctness Verification

**Test**: `tests/test_chunked_attention.py`
- 12 test cases (6 configs × 2 dtypes)
- All tests PASS with max error < 0.01
- float16: max_diff=0.000488, mean_diff~0.00001
- bfloat16: max_diff=0.003906, mean_diff~0.0001

### Key Files

- `nanovllm/kvcache/chunked_attention.py`: Triton kernels + merge function
- `tests/test_chunked_attention.py`: Correctness tests
- `tests/test_attention_offload.py`: Performance profiling

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 4096 | Tokens per block |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |

## Benchmarking

**Files**: `bench.py` (GPU), `bench_offload.py` (CPU offload), `bench_vllm.py` (comparison)

**Common Issues**:
1. `max_num_batched_tokens < max_model_len`: Set equal for long context
2. CUDA graph dimension mismatch: Ensure `input_len + output_len <= max_model_len`
3. RoPE out of bounds: Check model's `max_position_embeddings` in config.json

**Model Limits**:
- Qwen3-0.6B/4B: 40960 tokens
- Qwen2.5-7B-Instruct-1M: 1048576 tokens

**Performance (Qwen3-0.6B)**:
- GPU: ~18k tok/s (prefill), ~100 tok/s (decode)
- CPU Offload (16K): ~14k tok/s (prefill)
- CPU Offload (32K): ~13k tok/s (prefill)

## Performance Summary

### Completed Optimizations ✓

1. **sgDMA Integration** (2025-12-25)
   - Eliminated Device→Pageable transfers
   - Achieved 21-23 GB/s bandwidth (near PCIe limit)
   - 15.35x speedup on memory transfers

2. **Triton Fused Merge Kernel** (2025-12-25)
   - Reduced 7 PyTorch kernels → 2 Triton kernels
   - 4.3x speedup on merge operations
   - 1.67x overall ChunkedPrefill speedup

3. **N-way Pipeline with Dedicated Streams** (2025-12-25)
   - Per-slot transfer streams for parallel H2D across slots
   - Dedicated compute stream (avoids CUDA default stream implicit sync)
   - N-way pipeline using all available slots (not just 2-slot double buffering)
   - **2.0x improvement**: 7.2k → 14.1k tok/s (16K tokens prefill)

### Current Performance Bottlenecks

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

2. ~~**Pipeline Optimization**~~ ✓ COMPLETED
   - ~~Better overlap between compute and memory transfer~~
   - ~~Multi-stream execution~~
   - See: N-way Pipeline with Dedicated Streams above

3. **Alternative to sgDMA** (lower priority, PyTorch-only)
   - Reorganize cache layout: `[num_cpu_blocks, num_layers, ...]` instead of `[num_layers, num_cpu_blocks, ...]`
   - Trade-off: Extensive refactoring vs minimal sgDMA approach
   - Same performance as sgDMA (~24 GB/s)

---

**Author**: Zijie Tian
