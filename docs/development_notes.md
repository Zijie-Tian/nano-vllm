# Notes: Sparsity Integration into Layerwise Offload

## Current Architecture Analysis

### GPU-Only Path vs Offload Path

| Aspect | GPU-Only | Layerwise Offload |
|--------|----------|-------------------|
| KV Storage | GPU blocks (paged) | CPU pinned + GPU ring buffer |
| Prefill | All layers → then attention | Per-layer: attention → offload |
| Decode | FlashAttn with block table | Ring buffer H2D → FlashAttn |
| Sparse Support | MInference via `attention.py` | Not integrated |

### MInference Flow (GPU-Only)

```
attention.py:101-105:
  if context.sparse_prefill_policy is not None:
      o = context.sparse_prefill_policy.sparse_prefill_attention(q, k, v, layer_id)

minference.py:sparse_prefill_attention():
  1. estimate_pattern(q, k, layer_id) -> vertical_indices, slash_indices
  2. _triton_mixed_sparse_attention(q, k, v, indices)
  3. return output
```

### Quest Flow (GPU Block Mode)

```
hybrid_manager.py (if using CPU offload with Quest):
  select_blocks(available_blocks, ctx) -> selected block IDs
  -> load selected blocks to GPU
  -> standard FlashAttn with loaded blocks
```

### Layerwise Offload Prefill Flow

```
model_runner.py:run_layerwise_offload_prefill():
  for layer_id in range(num_layers):
      # QKV projection
      q, k, v = qkv_proj(hidden_ln)

      # RoPE
      q, k = rotary_emb(positions, q, k)

      # FULL attention (no sparsity!)
      attn_output = flash_attn_varlen_func(q, k, v, ...)

      # MLP
      hidden_states = mlp(attn_out + residual)

      # Sync offload ALL k, v to CPU
      for block_id in cpu_block_ids:
          k_cache_cpu[layer_id, block_id].copy_(k[start:end])
          v_cache_cpu[layer_id, block_id].copy_(v[start:end])
```

### Layerwise Offload Decode Flow

```
model_runner.py:run_layerwise_offload_decode():
  # Preload first N layers to ring buffer
  for i in range(num_buffers):
      offload_engine.load_layer_kv_to_buffer(i, i, cpu_block_table, valid_tokens)

  for layer_id in range(num_layers):
      current_buffer = layer_id % num_buffers

      # Wait for buffer load
      offload_engine.wait_buffer_load(current_buffer)

      # Get prefilled KV from ring buffer (ALL blocks loaded)
      k_prefill, v_prefill = offload_engine.get_buffer_kv(current_buffer, total_prefill_tokens)

      # QKV for new token
      q, k_new, v_new = qkv_proj(hidden_ln)

      # Concat and full attention
      k_full = torch.cat([k_prefill, k_decode_prev, k_new])
      attn_output = flash_attn_varlen_func(q, k_full, v_full, ...)

      # Start loading next layer
      offload_engine.load_layer_kv_to_buffer(current_buffer, layer_id + num_buffers, ...)
```

## Integration Points

### 1. Prefill Sparse Integration Point

**Location:** `model_runner.py:535-543`

**Current:**
```python
attn_output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=total_tokens,
    max_seqlen_k=total_tokens,
    softmax_scale=layer.self_attn.attn.scale,
    causal=True,
)
```

**After Integration:**
```python
if self.sparse_policy and self.sparse_policy.supports_offload_prefill:
    attn_output, k_sparse, v_sparse = self.sparse_policy.offload_prefill_attention(
        q, k, v, layer_id
    )
    k_to_offload = k_sparse if k_sparse is not None else k
    v_to_offload = v_sparse if v_sparse is not None else v
else:
    attn_output = flash_attn_varlen_func(q, k, v, ...)
    k_to_offload, v_to_offload = k, v
```

### 2. Decode Sparse Integration Point

**Location:** `model_runner.py:636-637` and `model_runner.py:704-706`

**Current (preload):**
```python
for i in range(num_preload):
    offload_engine.load_layer_kv_to_buffer(
        i, i, cpu_block_table, valid_tokens_per_block
    )
```

**After Integration:**
```python
for i in range(num_preload):
    layer_to_load = i
    if self.sparse_policy and self.sparse_policy.supports_offload_decode:
        # Prepare q for this layer (need to compute ahead)
        # OR: use previous layer's pattern as estimate
        selected_blocks = self.sparse_policy.select_offload_blocks(
            None,  # q not available yet at preload
            layer_to_load,
            cpu_block_table,
            valid_tokens_per_block
        )
    else:
        selected_blocks = cpu_block_table
    offload_engine.load_sparse_layer_kv_to_buffer(
        i, layer_to_load, selected_blocks, valid_tokens_per_block
    )
```

**Challenge:** Q is not available during preload phase!

**Solutions:**
1. Skip sparse preload, only sparse for non-preloaded layers
2. Use previous decode step's pattern as estimate
3. Add preload hook to sparse policy

### 3. Offload Engine Extension

**New Method in OffloadEngine:**

```python
def load_sparse_layer_kv_to_buffer(
    self,
    buffer_idx: int,
    layer_id: int,
    selected_cpu_block_ids: List[int],
    original_valid_tokens: List[int],
) -> int:
    """
    Load only selected blocks from CPU to buffer.

    Returns:
        Total tokens loaded (may be less than full sequence)
    """
    stream = self.layer_load_streams[buffer_idx]

    with torch.cuda.stream(stream):
        stream.wait_event(self.buffer_compute_done_events[buffer_idx])

        # Build mapping: original block -> selected position
        offset = 0
        for i, cpu_block_id in enumerate(selected_cpu_block_ids):
            # Find original index to get valid tokens
            valid_tokens = original_valid_tokens[i]  # Need mapping

            self.layer_k_cache[buffer_idx, offset:offset+valid_tokens].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id, :valid_tokens],
                non_blocking=True
            )
            # ... v_cache same

            offset += valid_tokens

        self.buffer_load_events[buffer_idx].record(stream)

    return offset  # Caller needs to know actual loaded tokens
```

## Metadata Flow for Quest

### During Prefill Offload

**Current:** No metadata collection in offload path

**Required:** Call `on_prefill_offload()` for each block

```python
# In run_layerwise_offload_prefill()
for i, cpu_block_id in enumerate(cpu_block_ids):
    start = i * block_size
    end = min(start + block_size, total_tokens)
    actual_size = end - start

    # BEFORE offload: update Quest metadata
    if self.sparse_policy and hasattr(self.sparse_policy, 'on_prefill_offload'):
        self.sparse_policy.on_prefill_offload(
            cpu_block_id, layer_id, k[start:end], actual_size
        )

    # Offload
    offload_engine.k_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(k[start:end])
    offload_engine.v_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(v[start:end])
```

### Quest Metadata Shape

```python
# BlockMetadataManager
key_min: [num_blocks, num_layers, num_kv_heads, head_dim]  # Min key per block per layer
key_max: [num_blocks, num_layers, num_kv_heads, head_dim]  # Max key per block per layer
```

**Memory:** 2 * num_blocks * num_layers * kv_heads * head_dim * 2 bytes
- Example: 1000 blocks * 28 layers * 4 heads * 128 dim * 2 * 2 = ~57 MB

## Performance Considerations

### MInference Prefill Overhead

| Operation | Time (64K seq) |
|-----------|----------------|
| Pattern estimation (last-64) | ~5ms |
| Triton sparse attention | ~80ms |
| Full FlashAttention | ~100ms |
| **Net Speedup** | ~15-20% |

### Quest Decode Overhead

| Operation | Time |
|-----------|------|
| Block scoring (GPU metadata) | ~0.1ms |
| Top-K selection | ~0.05ms |
| Sparse H2D load (8 blocks) | ~2ms |
| Full H2D load (100 blocks) | ~20ms |
| **Net Speedup** | ~10x H2D |

### Memory Trade-offs

| Mode | GPU Memory | CPU Memory | H2D Bandwidth |
|------|------------|------------|---------------|
| Full offload | Ring buffer | Full KV | High |
| Sparse offload | Ring buffer | Full KV | Low (subset) |
| Aggressive sparse | Ring buffer | Sparse KV | Very low |

## Edge Cases

### 1. Short Sequences (< sparse threshold)

```python
if total_tokens < sparse_threshold:
    # Fall back to full attention
    use_sparse = False
```

### 2. First Decode Step (no previous Q)

Quest can't score blocks without Q. Options:
- Use average embedding as proxy
- Load all blocks for first step
- Use prefill pattern as estimate

### 3. Variable Sequence Lengths in Batch

Layerwise offload currently only supports batch_size=1:
```python
assert len(seqs) == 1, "Layer-wise offload only supports single sequence"
```

Sparse integration should maintain this constraint.

### 4. Ring Buffer vs Sparse Load Mismatch

Ring buffer assumes fixed `total_prefill_tokens`:
```python
k_prefill, v_prefill = offload_engine.get_buffer_kv(buffer_idx, total_prefill_tokens)
```

Sparse load has variable token count. Need:
```python
# Track actual loaded tokens per buffer
loaded_tokens[buffer_idx] = sparse_load_count
k_prefill, v_prefill = offload_engine.get_buffer_kv(buffer_idx, loaded_tokens[buffer_idx])
```

## Testing Strategy

### Unit Tests

1. `test_sparse_policy_interface.py` - Verify new interface methods
2. `test_minference_offload.py` - MInference in offload mode
3. `test_quest_offload.py` - Quest block selection in offload mode

### Integration Tests

1. `test_offload_sparse_e2e.py` - Full prefill+decode with sparsity
2. `test_accuracy_comparison.py` - Compare outputs: full vs sparse

### Benchmarks

1. `bench_offload_sparse.py` - Compare:
   - Full offload (baseline)
   - MInference prefill + Quest decode
   - Aggressive sparse offload
