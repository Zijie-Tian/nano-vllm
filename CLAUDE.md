# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3 models with CPU offload for long-context inference.

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

**Pipeline**: Double buffering with `compute_done` events prevents data races. Pipeline depth = N-1 (prefill), (N-1)/2 (decode).

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

**Performance (Qwen3-0.6B, 40K)**:
- GPU: ~18k tok/s (prefill), ~100 tok/s (decode)
- CPU Offload: ~7.2k tok/s (prefill), ~3.5 tok/s (decode)

## TODO: Alternative Optimizations

### 1. Pure PyTorch Layout Reorganization (Alternative to sgDMA)

**Note**: sgDMA (above) already solves this. This is a pure-PyTorch alternative requiring more code changes.

**Change Layout**:
```python
# Current (non-contiguous access)
k_cache_cpu = torch.zeros(num_layers, num_cpu_blocks, block_size, kv_heads, head_dim,
                          pin_memory=True)
# Access: k_cache_cpu[:, block_id]  -> strided, slow

# Optimized (contiguous access)
k_cache_cpu = torch.zeros(num_cpu_blocks, num_layers, block_size, kv_heads, head_dim,
                          pin_memory=True)
# Access: k_cache_cpu[block_id]  -> contiguous, fast
```

**Files to Modify**:
- `kvcache/offload_engine.py`: Update all indexing in `load_to_slot_layer()`, `offload_slot_to_cpu()`
- Audit all `k_cache_cpu`/`v_cache_cpu` accesses

**Trade-off**:
- **sgDMA**: Minimal code changes, requires CUDA extension, 24.95 GB/s
- **Layout Change**: Pure PyTorch, extensive refactoring, 24.91 GB/s (same performance)

**Recommendation**: Use sgDMA for faster implementation with same performance.

---

**Author**: Zijie Tian
