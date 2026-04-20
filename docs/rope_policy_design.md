# RoPE Semantic Policy Design

> **Historical-scope note (2026-04-19):** this document explains the original PREROPE / POSTROPE semantic split and its alignment invariants. The **current Sparge-style sparse `POSTROPE` chunked-prefill implementation** is documented separately in [`docs/postrope_sparge_chunked_prefill_design.md`](postrope_sparge_chunked_prefill_design.md).

This document explains the design and invariants of the explicit `PREROPE` and `POSTROPE` policies.

## Why these policies exist

Before this change, nano-vLLM only had:

- `FULL`: the default full-attention implementation
- `TRIATTENTION`: a full-attention semantic variant that keeps Q/K in pre-RoPE form and applies RoPE lazily inside the policy

That was enough for implementation, but not enough for experiments. We needed:

1. an **explicit named policy for the existing post-RoPE/full-attention semantics**
2. an **explicit named policy for pre-RoPE KV-cache semantics**
3. a way to compare the two semantics with the same offload, chunking, and decode infrastructure

The result is:

- `POSTROPE`: explicit post-RoPE/full-attention semantics
- `PREROPE`: full-attention semantics with **pre-RoPE K-cache persistence**

## Semantic contract

| Policy | Attention input Q/K | KV cache / CPU offload stores | Where RoPE is applied |
|---|---|---|---|
| `FULL` | post-RoPE | post-RoPE K | in model attention block, before `Attention.forward()` |
| `POSTROPE` | post-RoPE | post-RoPE K | same as `FULL` |
| `PREROPE` | pre-RoPE | **pre-RoPE K** | lazily inside policy compute methods |
| `TRIATTENTION` | pre-RoPE | pre-RoPE K | lazily inside policy compute methods + decode-time token selection |

The critical invariant is:

> `PREROPE` must preserve **exactly the same generation semantics as `FULL`**, even though the cached K states are stored pre-RoPE instead of post-RoPE.

## Implementation layout

### Files

- `nanovllm/kvcache/sparse/prerope.py`
- `nanovllm/kvcache/sparse/postrope.py`
- `nanovllm/kvcache/sparse/__init__.py`
- `nanovllm/config.py`
- `nanovllm/engine/model_runner.py`

### Policy construction

Both `PREROPE` and `POSTROPE` are explicit `SparsePolicy` subclasses.

- `POSTROPE` is a semantic alias for the existing full-attention path
- `PREROPE` re-implements the full-attention flow while changing where RoPE is applied

### `apply_rope_in_attention`

`PREROPE` sets:

```python
apply_rope_in_attention = True
```

That causes model layers such as:

- `nanovllm/models/llama.py`
- `nanovllm/models/qwen2.py`
- `nanovllm/models/qwen3.py`
- `nanovllm/models/glm4.py`

to pass raw pre-RoPE Q/K into `Attention.forward(...)`.

`Attention.forward(...)` then places:

- `context.positions`
- `context.rotary_emb`

into the runtime context, and `PREROPE` applies RoPE at the last possible moment before calling the attention kernels.

## PREROPE data-flow

### Prefill

1. model produces raw pre-RoPE `q` / `k`
2. `Attention.forward()` stores raw `k` into the KV-cache path
3. `PREROPE.compute_prefill()` / `PREROPE.compute_chunked_prefill()` rotate:
   - current query tokens with current positions
   - historical/current keys with their own absolute positions
4. flash attention kernels consume the rotated tensors

### Decode

1. historical prefilled K blocks remain stored as pre-RoPE tensors in CPU offload storage
2. decode buffer stores the current step’s K in pre-RoPE form
3. `PREROPE.compute_chunked_decode()` materializes:
   - prefilled historical K
   - accumulated decode K
4. it then applies RoPE using the correct absolute positions before attention

## Important synchronization rule

The decode path exposed a subtle stream bug while validating `PREROPE`.

### Bug

`PREROPE._materialize_prefilled_cpu_history()` previously did:

1. `wait_slot_layer(slot)` on `compute_stream`
2. `get_kv_for_slot(slot)` on the default stream
3. clone/cat on the default stream

That is unsafe because `wait_slot_layer(slot)` only guarantees readiness on the stream that waited for the event. Reading the slot on the default stream can observe stale data.

### Fix

Historical slot reads now stay on `compute_stream`, and the method explicitly synchronizes back:

```python
torch.cuda.default_stream().wait_stream(compute_stream)
```

This keeps the PREROPE decode history materialization aligned with FULL semantics.

### Forward-looking rule

Any policy that:

- calls `wait_slot_layer(slot)`, then
- reads the slot tensor via `get_kv_for_slot(slot)`

must do both actions on the same stream, or explicitly bridge the streams before reading.

## Relationship to FULL and TRIATTENTION

### `POSTROPE` vs `FULL`

`POSTROPE` exists to make the semantics explicit in configs, tests, and experiment logs. It should be treated as the named post-RoPE/full-attention baseline.

### `PREROPE` vs `TRIATTENTION`

`PREROPE` and `TRIATTENTION` both keep pre-RoPE K in the cache, but they are not interchangeable:

- `PREROPE`: semantic-alignment policy; should match `FULL`
- `TRIATTENTION`: experimental policy; adds its own decode-time selection logic

## Validation summary

Validation was done in two layers:

1. **unit/shape-level validation**
   - `tests/test_dense_prefill_policy.py`
2. **behavioral alignment validation**
   - RULER runs under CPU offload + chunked prefill
   - exact output comparison for `FULL`, `POSTROPE`, and `PREROPE`

See:

- [`docs/ruler_rope_policy_alignment.md`](docs/ruler_rope_policy_alignment.md)
- [`docs/test_ruler_usage_guide.md`](docs/test_ruler_usage_guide.md)

## Practical takeaway

Use:

- `FULL` or `POSTROPE` when you want the existing post-RoPE baseline
- `PREROPE` when you want pre-RoPE cache semantics **without changing generation behavior**
- `TRIATTENTION` when you want the pre-RoPE cache semantics **plus** TriAttention-specific decode behavior
