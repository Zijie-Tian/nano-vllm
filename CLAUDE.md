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
- Blocks are 256 tokens by default

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

This enables:
1. Overlapped H2D transfer with attention computation
2. Each layer independently waits for its own data
3. Pipeline depth = N-1 for prefill (N slots, 1 for writing)

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

Manages both GPU and CPU blocks:
- `allocate()`: Allocate GPU block first, fallback to CPU
- `allocate_cpu_only()`: Force CPU allocation (for ring buffer mode)
- `get_all_cpu_blocks(seq)`: Get all CPU block IDs for a sequence
- `get_prefilled_cpu_blocks(seq)`: Get CPU blocks from previous chunks
- `get_write_slot_for_chunked_offload(seq)`: Get GPU slot for writing new KV (returns decode_slot)
- `may_offload()`: Offload GPU blocks to CPU when decode slot fills

### Online Softmax Merge

**File**: `nanovllm/kvcache/chunked_attention.py`

When computing attention across multiple chunks, results are merged using log-sum-exp:
```python
def merge_attention_outputs(o1, lse1, o2, lse2):
    # Uses LSE to correctly weight and combine partial attention outputs
```

### Pipeline Depth

- **Prefill**: Pipeline depth = N-1 (where N = num_gpu_blocks)
- **Decode**: Pipeline depth = (N-1)/2 (double buffering within decode_load_slots)
