# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3 models with CPU offload for long-context inference.

## GPU Mutex for Multi-Instance Debugging

**IMPORTANT**: When running multiple Claude instances for parallel debugging, only one GPU (cuda:0) is available. Before executing ANY command that uses the GPU (python scripts, benchmarks, tests), Claude MUST:

1. **Check GPU availability** by running:
   ```bash
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
   ```

2. **If processes are running on GPU**:
   - Wait and retry every 10 seconds until GPU is free
   - Use this polling loop:
     ```bash
     while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
       echo "GPU busy, waiting 10s..."
       sleep 10
     done
     ```

3. **Only proceed** when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` returns empty output

**Example workflow**:
```bash
# First check if GPU is in use
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader

# If output is empty, proceed with your command
python bench_offload.py

# If output shows processes, wait until they finish
```

**Note**: This applies to ALL GPU operations including:
- Running tests (`python tests/test_*.py`)
- Running benchmarks (`python bench*.py`)
- Running examples (`python example.py`)
- Any script that imports torch/cuda

## Local Package Installation for Multi-Instance

**CRITICAL**: After ANY code modification in the `nanovllm/` directory, you MUST reinstall the package before running tests or benchmarks:

```bash
pip install -e . --prefix=./.local --no-deps
```

Then run with PYTHONPATH:
```bash
PYTHONPATH=./.local/lib/python3.10/site-packages:$PYTHONPATH python <script.py>
```

**IMPORTANT**: When running multiple Claude instances on different worktrees, do NOT use `pip install -e .` globally as it will affect other instances. Instead, use local installation:

1. **Install to worktree-local directory**:
   ```bash
   pip install -e . --prefix=./.local --no-deps
   ```

2. **Set PYTHONPATH before running any Python command**:
   ```bash
   export PYTHONPATH=./.local/lib/python3.10/site-packages:$PYTHONPATH
   ```

3. **Combined example**:
   ```bash
   # One-liner for running tests with local package
   PYTHONPATH=./.local/lib/python3.10/site-packages:$PYTHONPATH python tests/test_needle.py
   ```

**Note**: The Python version in the path (python3.10) should match your environment.

## Sparse Attention

For sparse attention related content (block sparse attention, MInference, FlexPrefill, XAttention, AvgPool, etc.), refer to [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md).

### Quest Sparse Policy

**Files**: `nanovllm/kvcache/sparse/quest.py`, `nanovllm/kvcache/sparse/policy.py`

Quest policy selects Top-K blocks based on query-key similarity bounds using min/max key metadata.

**Scoring Mechanism**:
```python
score_min = torch.einsum('hd,bhd->bh', q, key_min)  # [num_blocks, kv_heads]
score_max = torch.einsum('hd,bhd->bh', q, key_max)  # [num_blocks, kv_heads]
scores = torch.maximum(score_min, score_max).mean(dim=-1)  # [num_blocks] ← averaged!
```

**Critical Limitation - No Per-Head Scheduling**:

The `.mean(dim=-1)` averages scores across all heads, making a **unified** block selection for all heads:

```
Block A: head0 needs (+4), head1 doesn't (-4) → avg = 0 → NOT selected
Block B: head0 doesn't (-4), head1 needs (+4) → avg = 0 → NOT selected
Block C: both heads moderately need (+2, +2) → avg = +2 → selected
```

**Why Per-Head Scheduling is Infeasible**:
1. **Memory Layout**: GPU cache stores all heads together `[block_size, kv_heads, head_dim]`
2. **FlashAttention**: Requires complete heads - partial heads cause dimension mismatch
3. **Block Granularity**: If any head needs a block, the entire block (all heads) must be loaded

**Policy Types**:
- `FullAttentionPolicy`: `supports_prefill=True, supports_decode=True` - loads all blocks
- `QuestPolicy`: `supports_prefill=False, supports_decode=True` - decode-only Top-K selection

## Architecture

### Core Components

- **LLMEngine** (`llm_engine.py`): Main entry, runs prefill-decode loop
- **ModelRunner** (`model_runner.py`): Loads weights, allocates KV cache, CUDA graphs
- **Scheduler** (`scheduler.py`): Two-phase scheduling (prefill → decode)
- **BlockManager** (`block_manager.py`): Paged attention with prefix caching (xxhash), default block size 4096
- **Attention** (`layers/attention.py`): FlashAttention with chunked methods for CPU offload

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

Key files:
- `tests/modeling_qwen3.py`: Reference Qwen3 implementation (torch + transformers only)
- `tests/test_needle_ref.py`: Reference needle test using custom Qwen3
- `tests/test_needle.py`: Needle-in-haystack test for nanovllm

### Common Pitfalls

1. **Shape mismatch**: nanovllm uses `[num_tokens, ...]` while torch uses `[batch, seq_len, ...]`
2. **Hook position**: `self_attn` captures after o_proj, `self_attn.attn` captures before o_proj
3. **Output format**: nanovllm returns tuple `(attn_output, None)`, handle with `output[0]`

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

### Key Files

- `nanovllm/kvcache/chunked_attention.py`: Triton kernels + merge function

## Known Issues and Fixes

### Partial Last Block Bug (FIXED ✓)

**Problem**: When prefill token count is not an exact multiple of `block_size`, decode outputs garbage.

**Root Cause**: `_chunked_decode_attention` calculated `last_block_valid_tokens` using `len(seq) - 1`, which increases during decode. But CPU blocks are fixed after prefill!

```python
# BUG: len(seq) increases each decode step
total_prefill_tokens = len(seq) - 1  # Wrong!
last_block_valid_tokens = total_prefill_tokens % block_size  # Reads garbage from CPU
```

**Fix**: Cache original prefill length in `HybridKVCacheManager.get_prefill_len()`:

```python
# CORRECT: Use cached prefill length
total_prefill_tokens = kvcache_manager.get_prefill_len(seq)  # Fixed value
```

**Files Modified**:
- `nanovllm/kvcache/hybrid_manager.py`: Added `_prefill_len` dict and `get_prefill_len()` method
- `nanovllm/layers/attention.py`: Use `get_prefill_len()` instead of `len(seq) - 1`

### Block Size 4096 Race Condition (FIXED ✓)

**Problem**: `block_size=4096` with multiple chunks produced `index_copy_(): index out of bounds` CUDA error during Chunk 2 processing.

**Root Cause**: Race condition between default stream and compute stream. In `_prepare_chunked_offload_chunk()`, `slot_mapping` tensor was created with `non_blocking=True` H2D transfer on the default stream. However, `store_kvcache` runs on `compute_stream`. Without synchronization, `compute_stream` could use `slot_mapping` before its transfer completed, causing corrupted indices.

**Fix** (in `attention.py`):
```python
if is_chunked_offload:
    compute_stream = context.kvcache_manager.offload_engine.compute_stream
    if k_cache.numel() and v_cache.numel():
        # CRITICAL: Wait for default stream to ensure slot_mapping tensor transfer is complete
        compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(compute_stream):
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

**Tested block sizes**: 512, 1024, 4096, 8192 - all pass.

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 1024 | Tokens per block (4096 now works after race condition fix) |
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
