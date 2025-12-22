# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Currently supports Qwen3 models.

## Architecture

### Core Components

**LLMEngine** (`nanovllm/engine/llm_engine.py`):
- Main entry point, wraps ModelRunner and Scheduler
- `generate()` runs prefill-decode loop until all sequences finish

**ModelRunner** (`nanovllm/engine/model_runner.py`):
- Loads model weights, allocates KV cache, captures CUDA graphs
- Rank 0 is main process; ranks 1+ run via `loop()` with shared memory events
- Chunked offload methods: `run_chunked_offload_prefill()`, `run_chunked_offload_decode()`

**Scheduler** (`nanovllm/engine/scheduler.py`):
- Two-phase scheduling: prefill (waiting queue) then decode (running queue)

**BlockManager** (`nanovllm/engine/block_manager.py`):
- Paged attention block allocation with prefix caching via xxhash
- Blocks are 4096 tokens by default (configurable via `kvcache_block_size`)

### Model & Attention

**Attention** (`nanovllm/layers/attention.py`):
- FlashAttention: `flash_attn_varlen_func` (prefill), `flash_attn_with_kvcache` (decode)
- Triton kernel `store_kvcache_kernel` for KV cache writes
- Chunked attention methods: `_chunked_prefill_attention()`, `_chunked_decode_attention()`

**Global Context** (`nanovllm/utils/context.py`):
- Stores attention metadata via `get_context()`/`set_context()`
- Key fields: `cu_seqlens`, `slot_mapping`, `block_tables`, `chunked_seq`, `kvcache_manager`
- `kvcache_manager`: Reference to HybridKVCacheManager for chunked attention (set when `is_chunked_prefill=True`)

## CPU Offload System

### Overview

When `enable_cpu_offload=True`, KV cache is stored on CPU with a small GPU buffer for computation. This enables long-context inference with limited GPU memory.

### Unified Ring Buffer Design

```
GPU Slots: [0]  [1]  [2]  [3]  [4]  ...
           ←────────────────────────────→
              All slots as ring buffer

Prefill: ALL slots cycle as ring buffer [slot = chunk_idx % N]
Decode:  slot[0] = decode_slot, slots[1:] = load slots for previous chunks
```

**File**: `nanovllm/kvcache/offload_engine.py`

Key attributes:
- `num_ring_slots`: Total GPU slots (= num_gpu_blocks)
- `ring_slots`: List of all GPU slot indices [0, 1, 2, ...]
- `decode_slot = 0`: Fixed slot for decode KV writes
- `decode_load_slots`: Slots[1:] for loading previous chunks during decode
- `k_cache_gpu/v_cache_gpu`: Shape `[num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]`
- `k_cache_cpu/v_cache_cpu`: Shape `[num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]` (pinned memory)

Key methods:
```python
# Prefill: get write slot and load slots
get_write_slot_for_prefill(chunk_idx)        # Returns chunk_idx % num_ring_slots
get_load_slots_for_prefill(write_slot_idx)   # Returns all slots except write_slot

# Decode: get load slots (excludes decode_slot)
get_load_slots_for_decode()                   # Returns slots[1:]

# Per-slot per-layer operations
load_to_slot_layer(slot_idx, layer_id, cpu_block_id)  # Async load single block
wait_slot_layer(slot_idx, layer_id)                   # Wait for layer's transfer
offload_slot_to_cpu(slot_idx, cpu_block_id)           # Async offload to CPU
```

### Per-Slot Per-Layer Events (Critical Design)

Each slot has per-layer CUDA events for fine-grained synchronization:
- `ring_slot_ready[slot_idx][layer_id]`: H2D transfer completion
- `ring_slot_offload_done[slot_idx][layer_id]`: D2H transfer completion
- `ring_slot_compute_done[slot_idx][layer_id]`: Attention compute completion (for safe buffer reuse)

This enables:
1. Overlapped H2D transfer with attention computation
2. Each layer independently waits for its own data
3. Pipeline depth = N-1 for prefill (N slots, 1 for writing)

### Async Pipeline with Double Buffering

**File**: `nanovllm/layers/attention.py` - `_ring_buffer_pipeline_load()`

The async pipeline uses double buffering with `compute_done` events to prevent data races:

```python
# Synchronization flow for safe async pipeline:
1. load_to_slot_layer() waits for compute_done[slot] before overwriting
2. wait_slot_layer() waits for slot_ready[slot] before reading
3. After flash_attn, record_slot_compute_done(slot) allows next load

Timeline with 2 slots (A, B):
┌──────────────┐
│ Load B0→A    │
└──────────────┘
               ┌──────────────┐ ┌──────────────┐
               │ Load B1→B    │ │ Load B2→A    │ ...
               └──────────────┘ └──────────────┘
                              ↘               ↘
                ┌──────────────┐ ┌──────────────┐
                │ Compute(A)   │ │ Compute(B)   │ ...
                └──────────────┘ └──────────────┘
```

**Key**: `load_to_slot_layer` internally waits for `compute_done` before starting transfer, preventing data race where new data overwrites unread data.

### Chunked Prefill Flow (Ring Buffer Pipeline)

**File**: `nanovllm/layers/attention.py` - `_chunked_prefill_attention()`

```
For prefill chunk K:
1. Current chunk's KV written to ring_slot[K % N]
2. Load previous chunks from CPU using N-1 available slots (pipeline)
3. Compute attention against previous KV (no causal mask)
4. Compute attention against current KV (causal mask)
5. Merge results using online softmax (LSE)
6. Offload current slot to CPU

Pipeline Timeline (with 4 slots, processing chunk 3):
write_slot = 3, load_slots = [0, 1, 2]

┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Load B0→S0   │ │Load B1→S1   │ │Load B2→S2   │ │   (wait)    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
              ↘               ↘               ↘
               ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
               │ Attn(B0)    │ │ Attn(B1)    │ │ Attn(B2)    │
               └─────────────┘ └─────────────┘ └─────────────┘
```

**Key**: Write slot cycles through ALL slots, load slots = all except write slot.

### Chunked Decode Flow (Double Buffering)

**File**: `nanovllm/layers/attention.py` - `_chunked_decode_attention()`

Decode uses legacy double-buffering with `decode_load_slots`:
- First half of decode_load_slots: 'compute' buffer
- Second half: 'prefetch' buffer

```
Timeline:
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
Load:   │C0 → buf0    │ │C1 → buf1    │ │C2 → buf0    │
        └─────────────┘ └─────────────┘ └─────────────┘
                      ↘               ↘               ↘
Compute:               [C0]           [C1]           [C2]

1. Pre-load first chunk to compute buffer
2. Wait for current buffer, trigger async prefetch to OTHER buffer
3. Compute attention, merge results
4. Swap buffers, repeat
5. Finally attend to decode_slot (new token's KV)
```

### HybridKVCacheManager

**File**: `nanovllm/kvcache/hybrid_manager.py`

CPU-primary KV cache manager with GPU ring buffer design:
- All KV cache is stored on CPU as primary storage
- GPU is used as a ring buffer for computation only
- Ring buffer enables pipelined H2D transfers overlapped with computation

Key methods:
- `allocate()` / `allocate_cpu_only()`: Allocate all blocks to CPU
- `get_all_cpu_blocks(seq)`: Get all CPU block IDs for a sequence
- `get_prefilled_cpu_blocks(seq)`: Get CPU blocks from previous chunks
- `get_write_slot_for_chunked_offload(seq)`: Get GPU slot for writing new KV (returns decode_slot)

### Online Softmax Merge

**File**: `nanovllm/kvcache/chunked_attention.py`

When computing attention across multiple chunks, results are merged using log-sum-exp:
```python
def merge_attention_outputs(o1, lse1, o2, lse2):
    # Uses LSE to correctly weight and combine partial attention outputs
```

### Flash Attention with LSE

**File**: `nanovllm/kvcache/chunked_attention.py` - `flash_attn_with_lse()`

Uses native `flash_attn_func` with `return_attn_probs=True` to get LSE output. This:
- Natively supports GQA (no memory overhead for head replication)
- Avoids `repeat_interleave` which would copy K/V heads (40MB+ per call)
- Returns `(output, lse)` for online softmax merging

### Pipeline Depth

- **Prefill**: Pipeline depth = N-1 (where N = num_gpu_blocks)
- **Decode**: Pipeline depth = (N-1)/2 (double buffering within decode_load_slots)

## Performance Optimizations

### Warmup Model Optimization

**File**: `nanovllm/engine/model_runner.py` - `warmup_model()`

Warmup uses a reasonable sequence length (`block_size * 2`) instead of `max_model_len`:
- Avoids huge intermediate activation memory allocation
- 8192 tokens is sufficient to trigger CUDA kernel JIT compilation
- Prevents OOM during initialization for long-context configs (256K+)

### Memory Considerations

**GQA Head Replication**: The chunked attention uses native `flash_attn_func` which handles GQA internally without memory overhead. Previous implementation used `repeat_interleave` which copied K/V heads, adding ~40MB per attention call.

**Block Size Trade-off**:
- Larger block_size (4096) = fewer H2D transfers, better throughput
- Smaller block_size (256) = finer granularity, less wasted memory
- Current default: 4096 tokens per block

## Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kvcache_block_size` | 4096 | Tokens per KV cache block |
| `max_num_batched_tokens` | 16384 | Max tokens per batch |
| `max_num_seqs` | 512 | Max concurrent sequences |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction for KV cache |
| `enforce_eager` | False | Disable CUDA graphs if True |
| `num_prefetch_blocks` | 2 | Ring buffer pipeline depth (deprecated, uses num_gpu_blocks) |

## Benchmarking

### Benchmark Files

| File | Purpose | Key Parameters |
|------|---------|----------------|
| `bench.py` | Standard GPU benchmark | Pure GPU inference |
| `bench_offload.py` | CPU offload benchmark | `enable_cpu_offload=True`, `num_gpu_blocks=8` |
| `bench_vllm.py` | vLLM comparison | Uses vLLM API for baseline comparison |

### Current Test Configuration

All benchmark files are aligned to use:
- **Model**: `~/models/Qwen3-0.6B/`
- **max_model_len**: 40960 (limited by model's `max_position_embeddings`)
- **Prefill test**: input_len = max_len - 1 (40959 tokens)
- **Decode test**: input_len = max_len - 128, output_len = 128

### Common Issues and Solutions

**1. `max_num_batched_tokens` assertion error**
```
AssertionError: assert self.max_num_batched_tokens >= self.max_model_len
```
**Solution**: Set `max_num_batched_tokens=max_model_len` when using large context lengths.

**2. CUDA graph block_tables dimension mismatch**
```
RuntimeError: The expanded size of the tensor (1) must match the existing size (2)
```
**Cause**: `input_len + output_len > max_model_len` causes more blocks than pre-allocated in CUDA graph.
**Solution**: Ensure `input_len + output_len <= max_model_len`.

**3. RoPE position embedding out of bounds**
```
Assertion `index out of bounds: 0 <= ... < 40960` failed
```
**Cause**: Sequence length exceeds model's `max_position_embeddings`.
**Solution**: Check model's `config.json` for `max_position_embeddings` and limit `max_model_len` accordingly.

### Model Context Length Limits

| Model | max_position_embeddings | Notes |
|-------|------------------------|-------|
| Qwen3-0.6B | 40960 | ~40K context |
| Qwen3-4B | 40960 | ~40K context |
| Qwen2.5-7B-Instruct-1M | 1048576 | 1M context |

**Important**: Always check `max_position_embeddings` in `config.json` before setting `max_model_len`.

### Performance Reference (Qwen3-0.6B, 40K context)

| Mode | Prefill (tok/s) | Decode (tok/s) |
|------|-----------------|----------------|
| GPU (bench.py) | ~18,000 | ~100 |
| CPU Offload (bench_offload.py) | ~7,200 | ~3.5 |

CPU offload trades performance for memory efficiency, enabling long-context inference on limited GPU memory.
