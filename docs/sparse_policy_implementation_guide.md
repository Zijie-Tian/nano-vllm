# SparsePolicy Implementation Guide

This guide describes how to implement a custom `SparsePolicy` for sparse attention in CPU offload mode.

## Overview

`SparsePolicy` is an abstract base class that controls:
1. **Block Selection**: Which KV cache blocks to load from CPU for each query
2. **Attention Computation**: How to compute chunked prefill and decode attention

All computation happens in the policy, with `attention.py` only delegating to the policy methods.

---

## Base Class Structure

```python
class SparsePolicy(ABC):
    # Phase support flags (REQUIRED to override)
    supports_prefill: bool = True
    supports_decode: bool = True

    # Abstract methods (MUST implement)
    def select_blocks(self, available_blocks, offload_engine, ctx) -> List[int]
    def compute_chunked_prefill(self, q, k, v, layer_id, ...) -> torch.Tensor
    def compute_chunked_decode(self, q, layer_id, ...) -> torch.Tensor

    # Optional hooks (CAN override)
    def initialize(self, num_layers, num_kv_heads, head_dim, num_cpu_blocks, dtype, device)
    def on_prefill_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens)
    def on_decode_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens)
    def reset(self)
```

---

## Required Implementations

### 1. Phase Support Flags

Every policy MUST declare which phases it supports:

```python
class MyPolicy(SparsePolicy):
    supports_prefill = True   # Can be used in prefill phase?
    supports_decode = True    # Can be used in decode phase?
```

| Policy Type | supports_prefill | supports_decode | Example |
|-------------|------------------|-----------------|---------|
| Full support | True | True | `FullAttentionPolicy` |
| Decode-only | False | True | `QuestPolicy` |
| Prefill-only | True | False | (hypothetical) |

### 2. select_blocks() - Block Selection

```python
@abstractmethod
def select_blocks(
    self,
    available_blocks: List[int],  # CPU block IDs with historical KV
    offload_engine: "OffloadEngine",
    ctx: PolicyContext,           # Context about current query
) -> List[int]:
    """Return subset of available_blocks to load."""
```

**PolicyContext fields:**
- `query_chunk_idx`: Current chunk index (0-indexed)
- `num_query_chunks`: Total number of chunks
- `layer_id`: Transformer layer index
- `query`: Query tensor (available for decode)
- `is_prefill`: True if prefill phase
- `block_size`: Tokens per block
- `total_kv_len`: Total KV length so far

**Example implementations:**

```python
# Full attention: load all blocks
def select_blocks(self, available_blocks, offload_engine, ctx):
    return available_blocks

# Top-K sparse: load K most important blocks
def select_blocks(self, available_blocks, offload_engine, ctx):
    scores = self.compute_block_scores(available_blocks, ctx.query)
    topk_indices = scores.topk(self.config.topk).indices
    return [available_blocks[i] for i in sorted(topk_indices.tolist())]
```

### 3. compute_chunked_prefill() - Prefill Attention

```python
@abstractmethod
def compute_chunked_prefill(
    self,
    q: torch.Tensor,              # [seq_len, num_heads, head_dim]
    k: torch.Tensor,              # [seq_len, num_kv_heads, head_dim] (unused)
    v: torch.Tensor,              # [seq_len, num_kv_heads, head_dim] (unused)
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    current_chunk_idx: int,
    seq: "Sequence",
    num_tokens: int,
) -> torch.Tensor:  # [seq_len, num_heads, head_dim]
```

**Required flow:**
1. Get historical blocks: `kvcache_manager.get_prefilled_cpu_blocks(seq)`
2. Call `select_blocks()` to filter blocks
3. Load blocks via ring buffer pipeline
4. Get current chunk KV: `offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)`
5. Compute attention with `flash_attn_with_lse()` (historical: causal=False, current: causal=True)
6. Merge results with `merge_attention_outputs()`
7. Return output with shape `[seq_len, num_heads, head_dim]`

**If policy doesn't support prefill:**
```python
def compute_chunked_prefill(self, ...):
    assert False, "MyPolicy does not support prefill phase"
```

### 4. compute_chunked_decode() - Decode Attention

```python
@abstractmethod
def compute_chunked_decode(
    self,
    q: torch.Tensor,              # [batch_size, num_heads, head_dim]
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    seq: "Sequence",
) -> torch.Tensor:  # [batch_size, 1, num_heads, head_dim]
```

**Required flow:**
1. Get prefilled blocks: `kvcache_manager.get_prefilled_cpu_blocks(seq)`
2. Calculate last block valid tokens from `kvcache_manager.get_prefill_len(seq)`
3. Call `select_blocks()` to filter blocks
4. Load blocks via `_decode_ring_buffer_pipeline()` helper
5. Read decode buffer: `offload_engine.decode_k_buffer[layer_id, ...]`
6. Merge results with `merge_attention_outputs()`
7. Return output with shape `[batch_size, 1, num_heads, head_dim]`

**If policy doesn't support decode:**
```python
def compute_chunked_decode(self, ...):
    assert False, "MyPolicy does not support decode phase"
```

---

## Optional Hooks

### initialize()

Called after KV cache allocation. Use to create metadata structures.

```python
def initialize(self, num_layers, num_kv_heads, head_dim, num_cpu_blocks, dtype, device):
    self.metadata = BlockMetadataManager(
        num_blocks=num_cpu_blocks,
        num_layers=num_layers,
        ...
    )
```

### on_prefill_offload() / on_decode_offload()

Called BEFORE GPU→CPU copy. Use to collect block metadata while data is still on GPU.

```python
def on_prefill_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens):
    # k_cache is still on GPU here
    self.metadata.update_min_max(cpu_block_id, layer_id, k_cache, num_valid_tokens)
```

### reset()

Called when starting new sequence. Use to clear state.

```python
def reset(self):
    if self.metadata is not None:
        self.metadata.reset()
```

---

## CPU-GPU Communication Rules

**MUST use OffloadEngine methods:**
```python
# Loading blocks
offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)
k, v = offload_engine.get_kv_for_slot(slot)
offload_engine.record_slot_compute_done(slot)

# Current chunk KV
k, v = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)

# Decode buffer
decode_k = offload_engine.decode_k_buffer[layer_id, start:end]
decode_v = offload_engine.decode_v_buffer[layer_id, start:end]
```

**NEVER do direct transfers:**
```python
# WRONG!
gpu_tensor.copy_(cpu_tensor)
gpu_tensor = cpu_tensor.to("cuda")
```

---

## Ring Buffer Pipeline Pattern

The standard pattern for loading blocks:

```python
def _decode_ring_buffer_pipeline(self, q_batched, cpu_block_table, load_slots, ...):
    from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

    num_blocks = len(cpu_block_table)
    num_slots = len(load_slots)
    o_acc, lse_acc = None, None

    # Phase 1: Pre-load up to num_slots blocks
    for i in range(min(num_slots, num_blocks)):
        offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

    # Phase 2: Process with pipeline
    for block_idx in range(num_blocks):
        slot = load_slots[block_idx % num_slots]

        # Wait for H2D transfer
        offload_engine.wait_slot_layer(slot)

        with torch.cuda.stream(offload_engine.compute_stream):
            # Get KV and compute attention
            k, v = offload_engine.get_kv_for_slot(slot)
            o, lse = flash_attn_with_lse(q_batched, k, v, softmax_scale, causal=False)
            offload_engine.record_slot_compute_done(slot)

        # Pipeline: start next block transfer
        next_idx = block_idx + num_slots
        if next_idx < num_blocks:
            offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_table[next_idx])

        # Merge results
        with torch.cuda.stream(offload_engine.compute_stream):
            if o_acc is None:
                o_acc, lse_acc = o, lse
            else:
                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o, lse)

    return o_acc, lse_acc
```

---

## Complete Example: Decode-Only Policy

```python
class TopKPolicy(SparsePolicy):
    """Load only top-K blocks based on query-key similarity."""

    supports_prefill = False  # Use FullAttentionPolicy for prefill
    supports_decode = True

    def __init__(self, topk: int = 8):
        self.topk = topk
        self.metadata = None

    def initialize(self, num_layers, num_kv_heads, head_dim, num_cpu_blocks, dtype, device):
        self.metadata = BlockMetadataManager(num_cpu_blocks, num_layers, num_kv_heads, head_dim)

    def select_blocks(self, available_blocks, offload_engine, ctx):
        if len(available_blocks) <= self.topk:
            return available_blocks

        # Compute scores and select top-K
        scores = self.metadata.compute_scores(available_blocks, ctx.layer_id, ctx.query)
        topk_indices = scores.topk(self.topk).indices.cpu().tolist()
        return [available_blocks[i] for i in sorted(topk_indices)]

    def on_prefill_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens):
        self.metadata.update(cpu_block_id, layer_id, k_cache, num_valid_tokens)

    def compute_chunked_prefill(self, ...):
        assert False, "TopKPolicy does not support prefill phase"

    def compute_chunked_decode(self, q, layer_id, softmax_scale, offload_engine, kvcache_manager, seq):
        # Copy implementation from FullAttentionPolicy.compute_chunked_decode
        # The only difference is select_blocks() will filter to top-K
        ...

    def reset(self):
        if self.metadata:
            self.metadata.reset()
```

---

## File Locations

| File | Purpose |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | Base class and PolicyContext |
| `nanovllm/kvcache/sparse/full_policy.py` | FullAttentionPolicy (reference implementation) |
| `nanovllm/kvcache/sparse/quest.py` | QuestPolicy (decode-only example) |
| `nanovllm/kvcache/chunked_attention.py` | `flash_attn_with_lse`, `merge_attention_outputs` |
