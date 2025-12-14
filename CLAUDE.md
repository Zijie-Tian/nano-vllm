# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Currently supports Qwen3 models.

## Commands

```bash
# Install
pip install -e .

# Run example
python example.py

# Run benchmarks
python bench.py                    # Standard benchmark
python bench_offload.py            # CPU offload benchmark

# Test chunked attention
CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 64 2
# Args: num_gpu_blocks input_len output_len num_prefetch_blocks
```

## Architecture

### Core Components

**LLMEngine** (`nanovllm/engine/llm_engine.py`):
- Main entry point, wraps ModelRunner and Scheduler
- `generate()` runs prefill-decode loop until all sequences finish

**ModelRunner** (`nanovllm/engine/model_runner.py`):
- Loads model weights, allocates KV cache, captures CUDA graphs
- Rank 0 is main process; ranks 1+ run via `loop()` with shared memory events

**Scheduler** (`nanovllm/engine/scheduler.py`):
- Two-phase scheduling: prefill (waiting queue) then decode (running queue)

**BlockManager** (`nanovllm/engine/block_manager.py`):
- Paged attention block allocation with prefix caching via xxhash
- Blocks are 256 tokens by default

### Model & Attention

**Attention** (`nanovllm/layers/attention.py`):
- FlashAttention: `flash_attn_varlen_func` (prefill), `flash_attn_with_kvcache` (decode)
- Triton kernel `store_kvcache_kernel` for KV cache writes
- Chunked attention methods: `_chunked_prefill_attention()`, `_chunked_decode_attention()`

**Global Context** (`nanovllm/utils/context.py`):
- Stores attention metadata via `get_context()`/`set_context()`
- Key fields: `cu_seqlens`, `slot_mapping`, `block_tables`, `chunked_seq`

## CPU Offload System

### Overview

When `enable_cpu_offload=True`, KV cache is stored on CPU with a small GPU buffer for computation. This enables long-context inference with limited GPU memory.

### Three-Region GPU Buffer Design

```
GPU Slots: [0]     [1, 2, 3]        [4, 5]
           ↑           ↑               ↑
        decode     compute         prefetch
        (1 slot)   (N slots)       (M slots)

- Decode slot: New token's KV written here during decode
- Compute region: Load CPU blocks for current chunk computation
- Prefetch region: Async load next chunk while computing current
```

**File**: `nanovllm/kvcache/offload_engine.py`

Key attributes:
- `decode_slot = 0`: Fixed slot for decode KV writes
- `compute_slots`: List of GPU slots for compute region
- `prefetch_slots`: List of GPU slots for prefetch region
- `k_cache_gpu/v_cache_gpu`: Shape `[num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]`
- `k_cache_cpu/v_cache_cpu`: Shape `[num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]` (pinned memory)

### Per-Layer Loading (Critical Design)

**Problem solved**: Original design had layer 0 load ALL layers' KV at once. When layer 0 processed chunk 1, it overwrote chunk 0's data before layer 1+ could read it.

**Solution**: Each layer independently loads only its own KV data:
```python
# Per-layer methods in OffloadEngine
load_to_compute_layer(layer_id, cpu_block_ids)  # Load single layer to compute region
wait_compute_layer(layer_id)                     # Wait for layer's transfer
load_to_prefetch_layer(layer_id, cpu_block_ids) # Load single layer to prefetch region
wait_prefetch_layer(layer_id)                    # Wait for layer's prefetch
```

### Chunked Prefill Flow

**File**: `nanovllm/layers/attention.py` - `_chunked_prefill_attention()`

```
For each prefill chunk:
1. Current chunk's KV is written to GPU (compute region slots)
2. Load previous chunks' KV from CPU to prefetch region
3. Compute attention against previous KV (no causal mask)
4. Compute attention against current KV (causal mask)
5. Merge results using online softmax (LSE)
6. Offload current chunk's KV to CPU
```

**Important**: Prefill uses ONLY prefetch region to avoid conflict with current chunk's KV being written to compute region.

### Chunked Decode Flow (Double Buffering)

**File**: `nanovllm/layers/attention.py` - `_chunked_decode_attention()`

```
Timeline (async double buffering):
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Load:   │C0 → Compute │ │C1 → Prefetch│ │C2 → Compute │
        └─────────────┘ └─────────────┘ └─────────────┘
                      ↘               ↘               ↘
Compute:               [C0]           [C1]           [C2]

1. Pre-load first chunk to compute region
2. Wait for current buffer, trigger async prefetch of next chunk to OTHER buffer
3. Compute attention, merge results
4. Swap buffers, repeat
5. Finally attend to decode slot (new token's KV)
```

### HybridKVCacheManager

**File**: `nanovllm/kvcache/hybrid_manager.py`

Manages both GPU and CPU blocks:
- `allocate()`: Allocate GPU block first, fallback to CPU
- `allocate_cpu_only()`: Force CPU allocation (for chunked prefill)
- `get_all_cpu_blocks(seq)`: Get all CPU block IDs for a sequence
- `get_prefilled_cpu_blocks(seq)`: Get CPU blocks from previous chunks
- `may_offload()`: Offload GPU blocks to CPU when decode slot fills

### Online Softmax Merge

**File**: `nanovllm/kvcache/chunked_attention.py`

When computing attention across multiple chunks, results are merged using log-sum-exp:
```python
def merge_attention_outputs(o1, lse1, o2, lse2):
    # Uses LSE to correctly weight and combine partial attention outputs
```

### Ring Buffer Design (Future Optimization)

Current double-buffering limits pipeline depth. Planned improvement:
- Unified ring buffer using all GPU slots (except decode)
- Per-slot per-layer CUDA events for fine-grained sync
- Deeper pipeline: prefetch N-1 blocks ahead (vs 1 chunk)

## Config Defaults

- `max_num_batched_tokens`: 16384
- `max_num_seqs`: 512
- `kvcache_block_size`: 256
- `gpu_memory_utilization`: 0.9
- `enforce_eager`: False (enables CUDA graphs)

## Testing CPU Offload

```bash
# Basic test with limited GPU blocks to trigger offload
CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 64 2

# Verify consistency (run multiple times, output should be identical)
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 32 2 2>&1 | tail -3
done
```
