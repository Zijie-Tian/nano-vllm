# Sparse Policy Integration with Layerwise Offload

This document describes the architecture and design of integrating sparse attention policies (MInference, Quest) with the layerwise CPU offload execution path.

## Design Goals

1. **Extend sparse policies to offload path**: GPU-only path already supports sparse policies, but layerwise offload bypasses them
2. **Maintain encapsulation**: All `copy_()` operations must be inside OffloadEngine, not exposed to model_runner
3. **Distinguish policy types**: Some policies affect attention computation (MInference), others affect KV load strategy (Quest)
4. **Extensible architecture**: Easy to add new sparse policies in the future

## Key Insight

The existing sparse policy implementation works, but the layerwise offload path bypasses it:

| Path | Attention Method | Sparse Support |
|------|------------------|----------------|
| GPU-only | `attention.py` → `sparse_prefill_attention()` | YES |
| Layerwise offload | `model_runner.py` → `flash_attn_varlen_func()` | NO (direct call) |

## Two Types of Sparse Policies

The fundamental difference between sparse policies:

| Policy | Affects Attention Computation | Affects KV Load Strategy | `select_blocks()` Behavior |
|--------|------------------------------|--------------------------|---------------------------|
| **MInference** | YES (`sparse_prefill_attention`) | NO | `return available_blocks` (all) |
| **Quest** | NO | YES | Returns Top-K subset |

- **MInference**: Only changes how attention is computed, doesn't affect external load/offload flow
- **Quest**: Selectively loads only some blocks, affects H2D transfer

## The `requires_block_selection` Interface Flag

To distinguish these policy types, we add a flag to the base class:

```python
# nanovllm/kvcache/sparse/policy.py
class SparsePolicy(ABC):
    # Existing flags
    supports_prefill: bool = True
    supports_decode: bool = True

    # NEW: Whether this policy requires selective block loading
    # If True: OffloadEngine will call select_blocks() before loading
    # If False: OffloadEngine will load all blocks (select_blocks ignored)
    requires_block_selection: bool = False
```

### Policy Implementations

```python
# MInference: prefill-only, no block selection
class MInferencePolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False
    requires_block_selection = False  # Only affects attention computation

# Quest: decode-only, requires block selection
class QuestPolicy(SparsePolicy):
    supports_prefill = False
    supports_decode = True
    requires_block_selection = True  # Affects KV load strategy

# Full attention: baseline
class FullAttentionPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = True
    requires_block_selection = False  # Load all blocks
```

## OffloadEngine Encapsulation

All KV cache operations are encapsulated in OffloadEngine. The model_runner never directly accesses internal storage.

### Prefill: Synchronous Offload with Hooks

```python
# nanovllm/kvcache/offload_engine.py
def offload_layer_kv_sync(
    self,
    layer_id: int,
    k: Tensor,
    v: Tensor,
    cpu_block_ids: List[int],
    total_tokens: int,
) -> None:
    """
    Synchronously offload layer KV to CPU.
    Calls sparse policy hooks internally.
    """
    for i, cpu_block_id in enumerate(cpu_block_ids):
        start = i * self.block_size
        end = min(start + self.block_size, total_tokens)
        actual_size = end - start

        # Hook: notify sparse policy BEFORE offload (k still on GPU)
        if self.sparse_policy is not None:
            self.sparse_policy.on_prefill_offload(
                cpu_block_id, layer_id, k[start:end], actual_size
            )

        # Synchronous copy to CPU (internal)
        self.k_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(k[start:end])
        self.v_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(v[start:end])
```

### Decode: Policy-Driven Block Loading

```python
def load_layer_kv_to_buffer_with_policy(
    self,
    buffer_idx: int,
    layer_id: int,
    cpu_block_ids: List[int],
    valid_tokens_per_block: List[int],
    query: Optional[Tensor] = None,
) -> int:
    """
    Load layer KV to buffer, optionally using sparse policy for block selection.

    Returns:
        Total tokens loaded
    """
    # Check if policy requires block selection
    if (self.sparse_policy is not None and
        self.sparse_policy.requires_block_selection and
        query is not None):
        # Build context
        ctx = PolicyContext(
            query_chunk_idx=0,
            num_query_chunks=1,
            layer_id=layer_id,
            query=query,
            is_prefill=False,
            block_size=self.block_size,
        )
        # Select blocks using policy
        selected_blocks = self.sparse_policy.select_blocks(cpu_block_ids, ctx)

        # Build valid_tokens for selected blocks
        block_to_valid = {bid: vt for bid, vt in zip(cpu_block_ids, valid_tokens_per_block)}
        selected_valid = [block_to_valid[bid] for bid in selected_blocks]

        return self._load_blocks_to_buffer(
            buffer_idx, layer_id, selected_blocks, selected_valid
        )
    else:
        # Load all blocks (no selection)
        return self._load_blocks_to_buffer(
            buffer_idx, layer_id, cpu_block_ids, valid_tokens_per_block
        )
```

## Prefill Integration (MInference)

MInference only affects attention computation, not the load/offload flow:

```python
# nanovllm/engine/model_runner.py - run_layerwise_offload_prefill()
def run_layerwise_offload_prefill(self, seqs):
    ...
    for layer_id in range(num_layers):
        # QKV projection + RoPE
        q, k = layer.self_attn.rotary_emb(positions, q, k)

        # Sparse or Full attention
        if self.sparse_prefill_policy is not None:
            # MInference: only changes attention computation
            attn_output = self.sparse_prefill_policy.sparse_prefill_attention(
                q, k, v, layer_id
            )
        else:
            # Full attention using FlashAttention
            attn_output = flash_attn_varlen_func(q, k, v, ...)

        # MLP
        ...

        # Offload ALL KV (MInference doesn't affect this)
        offload_engine.offload_layer_kv_sync(layer_id, k, v, cpu_block_ids, total_tokens)
```

### Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layerwise Offload Prefill                     │
│                      with MInference                             │
└─────────────────────────────────────────────────────────────────┘

For each layer:
┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐
│ QKV Proj     │───▶│ RoPE         │───▶│ sparse_prefill_attn()  │
│              │    │              │    │ (MInference pattern)   │
└──────────────┘    └──────────────┘    └───────────┬────────────┘
                                                    │
                    ┌──────────────┐    ┌───────────▼────────────┐
                    │ MLP          │◀───│ O Projection           │
                    │              │    │                        │
                    └──────┬───────┘    └────────────────────────┘
                           │
                    ┌──────▼───────┐
                    │ offload_     │    K, V still on GPU
                    │ layer_kv_    │───▶ Copy to CPU
                    │ sync()       │    (all blocks)
                    └──────────────┘
```

## Decode Integration (Quest - Infrastructure Ready)

Quest affects block load strategy. The infrastructure is ready, full integration deferred.

```python
# nanovllm/engine/model_runner.py - run_layerwise_offload_decode()
def run_layerwise_offload_decode(self, seqs):
    ...
    # Preload first N layers (no query available, full load)
    for i in range(num_preload):
        loaded_tokens[i] = offload_engine.load_layer_kv_to_buffer(
            i, i, cpu_block_table, valid_tokens_per_block
        )

    for layer_id in range(num_layers):
        current_buffer = layer_id % num_buffers

        # Wait for buffer load
        offload_engine.wait_buffer_load(current_buffer)

        # QKV projection
        q, k_new, v_new = ...

        # Get loaded KV from ring buffer
        k_prefill, v_prefill = offload_engine.get_buffer_kv(
            current_buffer, loaded_tokens[current_buffer]
        )

        # Attention
        ...

        # Mark buffer done
        offload_engine.record_buffer_compute_done(current_buffer)

        # Load next layer
        # Future: use load_layer_kv_to_buffer_with_policy(query=q) for Quest
        next_layer = layer_id + num_buffers
        if next_layer < num_layers:
            loaded_tokens[current_buffer] = offload_engine.load_layer_kv_to_buffer(
                current_buffer, next_layer, cpu_block_table, valid_tokens_per_block
            )
```

### Quest Integration (Future Work)

When Quest is fully integrated:

```python
# Load next layer with Quest block selection
if next_layer < num_layers:
    loaded_tokens[current_buffer] = offload_engine.load_layer_kv_to_buffer_with_policy(
        current_buffer, next_layer, cpu_block_table, valid_tokens_per_block,
        query=q  # Pass query for block selection
    )
```

**Challenge**: First N layers are preloaded before query is available, so they must use full load.

## Configuration

### Enabling Sparse Policy

```python
from nanovllm import LLM
from nanovllm.config import SparsePolicyType

# GPU-only with MInference
llm = LLM(
    model_path,
    sparse_policy=SparsePolicyType.MINFERENCE,
    minference_adaptive_budget=0.3,  # 30% of seq_len
)

# Offload with MInference
llm = LLM(
    model_path,
    enable_cpu_offload=True,
    num_gpu_blocks=2,
    sparse_policy=SparsePolicyType.MINFERENCE,
    minference_adaptive_budget=0.3,
)
```

### MInference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minference_adaptive_budget` | 0.3 | Budget as fraction of seq_len (0.3 = 30%) |
| `minference_vertical_size` | 1000 | Fixed vertical size (when budget=None) |
| `minference_slash_size` | 6096 | Fixed slash size (when budget=None) |
| `minference_num_sink_tokens` | 30 | Always-kept initial tokens |
| `minference_num_recent_diags` | 100 | Always-kept recent diagonals |

### Quest Parameters (for future decode integration)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sparse_topk_blocks` | 8 | Top-K blocks to load |
| `sparse_threshold_blocks` | 4 | Apply sparse only when blocks > threshold |

## Sparse Policy Hooks

Sparse policies can implement hooks for metadata collection:

```python
class SparsePolicy(ABC):
    def on_prefill_offload(
        self,
        block_id: int,
        layer_id: int,
        key: torch.Tensor,
        valid_tokens: int,
    ) -> None:
        """
        Hook called during prefill offload BEFORE KV is copied to CPU.
        Key tensor is still on GPU - can compute metadata efficiently.

        Used by Quest to compute min/max key statistics for block selection.
        """
        pass

    def on_decode_offload(
        self,
        block_id: int,
        keys: torch.Tensor,  # [num_layers, block_size, kv_heads, head_dim]
    ) -> None:
        """
        Hook called when decode buffer is offloaded to CPU.
        """
        pass
```

## File Changes Summary

| File | Changes |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | Add `requires_block_selection` attribute |
| `nanovllm/kvcache/sparse/minference.py` | Set `requires_block_selection = False` |
| `nanovllm/kvcache/sparse/quest.py` | Set `requires_block_selection = True` |
| `nanovllm/kvcache/sparse/full_policy.py` | Set `requires_block_selection = False` |
| `nanovllm/kvcache/offload_engine.py` | Add `offload_layer_kv_sync()`, sparse hooks |
| `nanovllm/engine/model_runner.py` | Integrate sparse policies in offload paths |

## Key Design Principles

1. **Encapsulation**: All `copy_()` operations inside OffloadEngine
2. **Interface Flag**: `requires_block_selection` declares policy type
3. **Separation of Concerns**:
   - MInference: only `sparse_prefill_attention()` (compute-level)
   - Quest: `select_blocks()` + hooks (load-level)
4. **Hooks Inside Engine**: Policy hooks called within OffloadEngine methods

## Test Results

Verified on Qwen3-4B-Instruct-2507 with 32K input:

```
# GPU-only + MInference
test_needle.py --model Qwen3-4B --input-len 32768 --enable-minference
- Prefill: 3383 tok/s
- Output: "7492<|im_end|>"
- Result: PASSED

# Offload + MInference
test_needle.py --model Qwen3-4B --input-len 32768 --enable-offload --enable-minference
- Prefill: 5373 tok/s
- Output: "7492<|im_end|>"
- Result: PASSED
```

Both configurations produce identical outputs, confirming correctness.

## Related Documents

- [`sparse_attention_guide.md`](sparse_attention_guide.md): Algorithm details for sparse methods
- [`architecture_guide.md`](architecture_guide.md): Overall system architecture
- [`gpu_only_performance_issue.md`](gpu_only_performance_issue.md): Why offload is faster than GPU-only
