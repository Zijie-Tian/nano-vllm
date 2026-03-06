# GEMINI.md (Sparse Policies)

## Sparse Policy Mandates (CRITICAL)

### 1. No `None` Policy
- `sparse_policy` parameter **must never be `None`**.
- It must at least be an instance of `FullAttentionPolicy`.
- Explicitly handle `None` in constructors/configs by defaulting to `SparsePolicyType.FULL`.

### 2. Base Class Requirements
- Must inherit from `SparsePolicy`.
- Must declare `supports_prefill` and `supports_decode` (boolean).
- If a phase is not supported, `assert False, "Phase not supported"` must be used in the corresponding `compute_chunked_*` method.

### 3. CPU-GPU Communication
- **PROHIBITED**: `torch.Tensor.copy_()` or `.to(device)` directly in compute methods.
- **MANDATORY**: Use `OffloadEngine` methods for all transfers:
    - `offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)`
    - `offload_engine.wait_slot_layer(slot)`
    - `offload_engine.get_kv_for_slot(slot)`
- This ensures correct stream synchronization and pipeline performance.

### 4. Method Signatures
- `select_blocks(available_blocks, offload_engine, ctx) -> List[int]`
- `compute_chunked_prefill(q, k, v, layer_id, softmax_scale, offload_engine, kvcache_manager, current_chunk_idx, seq, num_tokens) -> torch.Tensor`
- `compute_chunked_decode(q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq) -> torch.Tensor`
