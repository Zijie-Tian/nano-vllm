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

# Run benchmark
python bench.py
```

## Architecture

### Core Components

**LLMEngine** (`nanovllm/engine/llm_engine.py`):
- Main entry point, wraps ModelRunner and Scheduler
- Handles tokenization and multi-process tensor parallelism coordination
- `generate()` method runs the prefill-decode loop until all sequences finish

**ModelRunner** (`nanovllm/engine/model_runner.py`):
- Loads model weights, allocates KV cache, captures CUDA graphs
- Rank 0 is the main process; ranks 1+ run in separate processes via `loop()` waiting on shared memory events
- `run()` prepares inputs and executes model forward pass

**Scheduler** (`nanovllm/engine/scheduler.py`):
- Two-phase scheduling: prefill (waiting queue) then decode (running queue)
- Handles preemption when memory is constrained by moving sequences back to waiting

**BlockManager** (`nanovllm/engine/block_manager.py`):
- Paged attention block allocation with prefix caching via xxhash
- Blocks are 256 tokens by default, tracked with reference counting

**Sequence** (`nanovllm/engine/sequence.py`):
- Tracks token IDs, block table, and sampling parameters per request
- Custom `__getstate__`/`__setstate__` for efficient pickling across processes

### Model Implementation

**Qwen3ForCausalLM** (`nanovllm/models/qwen3.py`):
- Standard transformer: embedding → decoder layers → RMSNorm → LM head
- Uses `packed_modules_mapping` for weight loading (q/k/v → qkv_proj, gate/up → gate_up_proj)

**Attention** (`nanovllm/layers/attention.py`):
- Uses FlashAttention (`flash_attn_varlen_func` for prefill, `flash_attn_with_kvcache` for decode)
- Custom Triton kernel `store_kvcache_kernel` for KV cache writes

**Parallel Layers** (`nanovllm/layers/linear.py`, `embed_head.py`):
- Tensor parallelism via column/row parallel linear layers with custom weight loaders

### Key Design Patterns

- **Global Context**: `nanovllm/utils/context.py` stores attention metadata (cu_seqlens, slot_mapping, block_tables) accessed via `get_context()`/`set_context()`
- **CUDA Graph Capture**: Decode phase uses captured graphs for batch sizes 1, 2, 4, 8, 16, 32... up to max_num_seqs (capped at 512)
- **Shared Memory IPC**: Tensor parallel workers receive commands via pickled data in SharedMemory, synchronized with Events

### Config Defaults

- `max_num_batched_tokens`: 16384
- `max_num_seqs`: 512
- `kvcache_block_size`: 256
- `gpu_memory_utilization`: 0.9
- `enforce_eager`: False (enables CUDA graphs)
