# COMPASS Head-First KV Layout & Async Staging Buffer Guide

This document records two critical architectural optimizations made to the KV Cache and `COMPASSPolicy` for extreme long-context (e.g., 32K+ tokens) sparse attention offloading. These two changes combined allow COMPASS prefill to execute efficiently without overwhelming H2D amplification and CPU synchronous blocking overhead.

## 1. Head-First KV Cache Layout (`is_head_first`)

### The Problem: H2D PCIe Amplification
Historically, `OffloadEngine` allocated CPU/GPU cache buffers in the shape `[..., block_size, num_kv_heads, head_dim]`. 
When `COMPASSPolicy` performed per-head sub-block gathering, it would slice `staging_k_cpu[:num_tokens]`. However, memory slice operations with interleaved heads mean that transferring $N$ tokens for **one head** inherently dragged along the memory for **all heads** under the hood. For a model with 8 KV heads, transferring 1 valid head’s data caused a massive 8x PCIe bandwidth waste (~180 GB H2D transfer for a 32K context instead of ~22 GB).

### The Solution: `[..., num_kv_heads, block_size, head_dim]`
To eliminate this amplification, `OffloadEngine` introduced the `is_head_first` layout flag. 
When `sparse_policy` is `COMPASS`, the `OffloadEngine` dynamically transposes its internal buffer shapes for both `k_cache_gpu`, `v_cache_gpu` and `staging_k_cpu`, `staging_v_cpu` to prioritize the head dimension: `[..., num_kv_heads, block_size, head_dim]`.

*   **Gather Operation**: `gather_subblocks_per_head` was adapted to write contiguous data along the token dimension:
    ```python
    staging_k[staging_idx, head_idx, :num_tokens, :].copy_(k_block_data)
    ```
*   **H2D Transfer**: `load_staging_to_slot` now slices out memory for exactly one head at a time, resulting in a single contiguous PCIe transaction with zero waste.
    ```python
    k_cache_gpu[slot_idx, head_idx, :num].copy_(staging_k[head_idx, :num], non_blocking=True)
    ```
*   **Backward Compatibility**: All existing functions (like `get_layer_cache` for decoding, `get_prefill_buffer_slice` for FallBack attention) dynamically append `.transpose(1, 2).contiguous()` to safely map the Head-First layout back to the standard layout expected by Torch/Triton compute kernels without mutating original code.

---

## 2. Multi-Staging Async Pipeline

### The Problem: CPU `cuda.sync` Bottleneck
Even after `gather_subblocks_per_head` became coalesced, `COMPASSPolicy` still blocked the CPU inside the prefill loop. The CPU had to wait for the GPU H2D transfer to finish in `load_staging_to_slot` (`torch.cuda.current_stream().synchronize()`) before it could overwrite `staging_k_cpu` and `staging_v_cpu` with the subsequent token batch. 
This CPU-blocking pattern forced the GPU BLASST kernels into starvation during the gather phase.

### The Solution: Rotating N-Staging Buffers
`OffloadEngine` now allocates multiple sets of pinned memory staging buffers based on `num_staging_buffers` (defaults to 8).

1.  **Index Rotation**: Inside `COMPASSPolicy.compute_chunked_prefill`, we rotate through the available staging buffers:
    ```python
    staging_idx = batch_idx % offload_engine.num_staging_buffers
    ```
2.  **Per-Buffer Synchronization**: Instead of a global GPU sync, we associate a `torch.cuda.Event` (`staging_ready_events`) with each staging buffer.
    Before writing to `staging_k_cpu[staging_idx]`, the CPU waits for that specific buffer's event to be recorded by the transfer stream. Since $N > 1$, the CPU can freely write to buffer $[N+1]$ while the GPU is still consuming buffer $[N]$.
3.  **Result**: The global CPU-GPU synchronization in `load_staging_to_slot` was entirely removed (`_sync_after_load` set to `False`). The gathering process on CPU runs completely asynchronously and decoupled from PCIe transfers and BLASST GPU computation.

### Profiling Confirmations
Both changes were exhaustively profiled via Nsight Systems:
*   **H2D Size**: Reduced from 187+ GB down to exactly matching algorithmic expected bytes (~22GB) for 32K context.
*   **Prefill Pipeline**: Eliminated the `staging_sync` NVTX block wait, enabling overlapped PCIe execution and maximizing GPU compute utilization. The dominant bottleneck has now properly shifted to algorithmic Python CPU overheads (`cuda.sync` waiting on GPU kernels ahead of time, Top-P tensor ops, etc.).
