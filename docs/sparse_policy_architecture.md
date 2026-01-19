# SparsePolicy Architecture Guide

This document describes the SparsePolicy abstraction for chunked attention computation in CPU offload mode.

## Overview

SparsePolicy is an abstract base class that defines how attention is computed during chunked prefill and decode phases. All attention computation logic is delegated to the policy, allowing different sparse attention strategies to be implemented without modifying the core attention layer.

```
attention.py                     SparsePolicy
    |                                 |
    | _chunked_prefill_attention      |
    | ────────────────────────────>   | compute_chunked_attention()
    |                                 |
    | _chunked_decode_attention       |
    | ────────────────────────────>   | compute_chunked_decode()
    |                                 |
```

## Key Design Principles

1. **Delegation Pattern**: `attention.py` only validates and delegates; all computation is in the policy
2. **No Direct Imports**: `attention.py` does not import `flash_attn_with_lse` or `merge_attention_outputs`
3. **Pipeline Encapsulation**: Ring buffer and cross-layer pipelines are internal to the policy
4. **Phase Support Flags**: Policies declare which phases they support via `supports_prefill` and `supports_decode`

---

## SparsePolicy Base Class

**File**: `nanovllm/kvcache/sparse/policy.py`

### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `supports_prefill` | bool | Whether policy supports prefill phase |
| `supports_decode` | bool | Whether policy supports decode phase |

### Abstract Methods

```python
@abstractmethod
def select_blocks(
    self,
    available_blocks: List[int],
    offload_engine: "OffloadEngine",
    ctx: PolicyContext,
) -> List[int]:
    """Select which KV blocks to load for the current query chunk."""
    pass

@abstractmethod
def compute_chunked_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    current_chunk_idx: int,
    seq: "Sequence",
    num_tokens: int,
) -> torch.Tensor:
    """Compute chunked prefill attention (complete flow)."""
    pass

@abstractmethod
def compute_chunked_decode(
    self,
    q: torch.Tensor,
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    seq: "Sequence",
) -> torch.Tensor:
    """Compute chunked decode attention (complete flow)."""
    pass
```

### Hook Methods

| Method | When Called | Purpose |
|--------|-------------|---------|
| `initialize()` | After KV cache allocation | Initialize policy resources (e.g., metadata) |
| `on_prefill_offload()` | Before GPU→CPU copy during prefill | Collect block metadata |
| `on_decode_offload()` | Before GPU→CPU copy during decode | Update block metadata |
| `reset()` | New sequence / clear state | Reset policy state |

---

## FullAttentionPolicy

**File**: `nanovllm/kvcache/sparse/full_policy.py`

The default policy that loads all blocks (no sparsity). Serves as the baseline implementation.

### Flags

```python
supports_prefill = True
supports_decode = True
```

### Prefill Flow (`compute_chunked_attention`)

```
1. Get historical blocks from kvcache_manager
   └── cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

2. Apply select_blocks (returns all for FullPolicy)
   └── cpu_block_table = self.select_blocks(cpu_block_table, offload_engine, ctx)

3. Load and compute historical blocks via ring buffer
   └── For each block:
       a. load_to_slot_layer(slot, layer_id, cpu_block_id)
       b. wait_slot_layer(slot)
       c. prev_k, prev_v = get_kv_for_slot(slot)
       d. flash_attn_with_lse(q, prev_k, prev_v, causal=False)
       e. merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)

4. Compute current chunk attention (causal)
   └── k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
   └── flash_attn_with_lse(q, k_curr, v_curr, causal=True)

5. Merge historical and current attention
   └── merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)
```

### Decode Flow (`compute_chunked_decode`)

```
1. Get prefilled CPU blocks
   └── cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)

2. Calculate last block valid tokens
   └── total_prefill_tokens = kvcache_manager.get_prefill_len(seq)
   └── last_block_valid_tokens = total_prefill_tokens % block_size

3. Apply select_blocks for block filtering
   └── cpu_block_table = self.select_blocks(cpu_block_table, offload_engine, ctx)

4. Load prefilled blocks via pipeline
   └── IF is_pipeline_active():
       └── _decode_with_layer_pipeline()  # Cross-layer pipeline
   └── ELSE:
       └── _decode_ring_buffer_pipeline()  # Ring buffer fallback

5. Read accumulated decode tokens from decode buffer
   └── decode_k = offload_engine.decode_k_buffer[layer_id, start:end]
   └── decode_v = offload_engine.decode_v_buffer[layer_id, start:end]
   └── flash_attn_with_lse(q, decode_k, decode_v, causal=False)

6. Merge all results
   └── merge_attention_outputs(o_acc, lse_acc, decode_o, decode_lse)
```

---

## Pipeline Modes

### Ring Buffer Pipeline (`_decode_ring_buffer_pipeline`)

Used when cross-layer pipeline is not active. Loads blocks one by one using ring buffer slots.

```
Slot[0]: Block A ──> Compute ──> Block C ──> Compute
Slot[1]: Block B ──> Compute ──> Block D ──> Compute
```

**Advantages**:
- Simple, proven correctness
- Works with any number of slots

**Flow**:
```python
# Phase 1: Pre-load up to num_slots blocks
for i in range(num_preload):
    offload_engine.load_to_slot_layer(load_slots[i], layer_id, cpu_block_table[i])

# Phase 2: Process blocks with pipeline
for block_idx in range(num_blocks):
    current_slot = load_slots[block_idx % num_slots]

    # Wait for transfer
    offload_engine.wait_slot_layer(current_slot)

    # Compute attention
    prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
    prev_o, prev_lse = flash_attn_with_lse(q, prev_k, prev_v, causal=False)
    offload_engine.record_slot_compute_done(current_slot)

    # Pipeline: start loading next block
    if next_block_idx < num_blocks:
        offload_engine.load_to_slot_layer(current_slot, layer_id, cpu_block_table[next_block_idx])

    # Merge results
    o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
```

### Cross-Layer Pipeline (`_decode_with_layer_pipeline`)

Optimized for decode when all layers need the same blocks. Uses double-buffered layer cache.

```
Layer 0: Wait Layer 0 ──> Compute ──> Trigger Layer 1 load
Layer 1: Wait Layer 1 ──> Compute ──> Trigger Layer 2 load
Layer 2: Wait Layer 2 ──> Compute ──> ...
```

**Advantages**:
- Overlaps H2D transfer with computation across layers
- Reduces effective latency: O(transfer + layers × compute) vs O(layers × transfer)

**Flow**:
```python
# Get KV from pre-loaded layer buffer (triggers next layer loading)
prev_k, prev_v = offload_engine.get_decode_layer_kv(layer_id, num_blocks)

# Reshape for FlashAttention
# prev_k, prev_v: [num_blocks, block_size, kv_heads, head_dim]
#              -> [1, total_tokens, kv_heads, head_dim]

# Handle partial last block
if last_block_valid_tokens < block_size:
    actual_tokens = (num_blocks - 1) * block_size + last_block_valid_tokens
    prev_k_flat = prev_k.reshape(-1, kv_heads, head_dim)[:actual_tokens]

# Compute attention on all prefilled blocks at once
o_acc, lse_acc = flash_attn_with_lse(q, prev_k_batched, prev_v_batched, causal=False)
```

---

## Code Conventions

### Unsupported Phases Must Assert False

If a policy doesn't support a phase, the corresponding method must `assert False`:

```python
class PrefillOnlyPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False

    def compute_chunked_attention(self, ...):
        # Normal prefill implementation
        ...

    def compute_chunked_decode(self, ...):
        assert False, "PrefillOnlyPolicy does not support decode phase"
```

### Caller Must Check Support Flags

`attention.py` checks support flags before calling:

```python
if not sparse_policy.supports_decode:
    raise RuntimeError(f"{sparse_policy} does not support decode phase")
```

This provides double protection:
1. Caller check → Clear error message
2. Method assert → Prevents bypassing the check

### CPU-GPU Communication via OffloadEngine Only

All CPU-GPU data transfers must go through `OffloadEngine` methods:

```python
# Correct: Use OffloadEngine methods
offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)
k, v = offload_engine.get_kv_for_slot(slot)

# Incorrect: Direct torch operations
gpu_tensor.copy_(cpu_tensor)  # DON'T DO THIS
gpu_tensor = cpu_tensor.to("cuda")  # DON'T DO THIS
```

---

## File Structure

| File | Purpose |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | Base class, PolicyContext, abstract methods |
| `nanovllm/kvcache/sparse/full_policy.py` | FullAttentionPolicy implementation |
| `nanovllm/kvcache/sparse/quest.py` | QuestPolicy (decode-only Top-K selection) |
| `nanovllm/layers/attention.py` | Attention layer, delegates to policy |

---

## Policy Implementations

| Policy | supports_prefill | supports_decode | Description |
|--------|------------------|-----------------|-------------|
| `FullAttentionPolicy` | True | True | Loads all blocks (baseline) |
| `QuestPolicy` | False | True | Decode-only Top-K selection |
| `XAttentionBSAPolicy` | False | False | Placeholder for future BSA |

---

## Testing

Run needle-in-haystack test with offload:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload
```

Expected output:
```
Needle-in-Haystack Test
Model: Llama-3.1-8B-Instruct
CPU offload: True
Sparse policy: FULL
Result: PASSED
```
