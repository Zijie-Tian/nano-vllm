# Architecture Guide

This document describes the core components and design of nano-vLLM, with detailed focus on the CPU offload system.

## Core Components

### LLMEngine (`llm_engine.py`)
Main entry point that runs the prefill-decode loop. Manages the overall inference workflow.

### ModelRunner (`model_runner.py`)
- Loads model weights
- Allocates KV cache
- Manages CUDA graphs for decode acceleration

### Scheduler (`scheduler.py`)
Two-phase scheduling system:
- **Prefill phase**: Processes prompt tokens
- **Decode phase**: Generates output tokens autoregressively

### BlockManager (`block_manager.py`)
- Paged attention implementation
- Prefix caching using xxhash
- Default block size: 4096 tokens

### Attention (`layers/attention.py`)
- FlashAttention for efficient computation
- Chunked methods for CPU offload mode

---

## CPU Offload System

### Ring Buffer Design

The CPU offload system uses a unified ring buffer to manage GPU memory slots:

```
GPU Slots: [0]  [1]  [2]  [3]  ...  (unified ring buffer)
Prefill:  slot = chunk_idx % N
Decode:   slot[0] = decode, slots[1:] = load previous chunks
```

**Key Files**: `kvcache/offload_engine.py`, `kvcache/hybrid_manager.py`

### Memory Layout

**GPU Memory**:
```
[num_layers, num_gpu_blocks, block_size, kv_heads, head_dim]
```

**CPU Memory** (pinned):
```
[num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `load_to_slot_layer(slot, layer, cpu_block)` | Async H2D load for specific layer |
| `offload_slot_to_cpu(slot, cpu_block)` | Async D2H offload |
| Per-slot per-layer CUDA events | Fine-grained synchronization |

### Pipeline Architecture

**N-way Pipeline** with dedicated streams for full compute-transfer overlap:

- **Prefill pipeline depth**: N-1
- **Decode pipeline depth**: (N-1)/2

### Stream Architecture

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

### Chunked Offload Flow

**Prefill Phase**:
1. For each chunk, assign `slot = chunk_idx % N`
2. Load required KV blocks from CPU to assigned slot
3. Compute attention on current chunk
4. Offload results back to CPU if needed

**Decode Phase**:
1. Use `slot[0]` for active decode computation
2. Use `slots[1:]` to prefetch upcoming chunks
3. Rotate slots as decoding progresses

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kvcache_block_size` | 1024 | Tokens per KV cache block |
| `num_gpu_blocks` | 2 | Number of GPU blocks for offload |
| `num_kv_buffers` | 4 | Ring buffer size (1-4), lower = less memory but slower decode |
| `enable_cpu_offload` | False | Enable CPU offload mode |

### Trade-offs

- **More GPU blocks**: Higher memory usage, faster prefill (fewer transfers)
- **Fewer GPU blocks**: Lower memory usage, more frequent transfers
- **Larger ring buffer**: More memory, better prefetch overlap
- **Smaller ring buffer**: Less memory, potential compute stalls

---

**Author**: Zijie Tian
