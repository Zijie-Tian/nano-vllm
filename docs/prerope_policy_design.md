# PreRoPE Policy Design

## Overview

**PreRoPE** is a `SparsePolicy` that preserves **exactly the same generation semantics as `FULL` attention**, while changing the internal storage format of the KV cache from post-RoPE to **pre-RoPE**. The key idea is **lazy RoPE application**: RoPE rotation is deferred from the model layer to the policy's compute methods, applied at the last possible moment before the attention kernel runs.

This design unlocks two important capabilities:

1. **Pre-RoPE KV cache persists raw amplitude/phase invariants**, enabling novel sparse attention strategies (e.g., QPOOL) that reason about post-RoPE attention contributions without ever rotating a vector.
2. **Semantic equivalence with FULL** allows PreRoPE to serve as a drop-in replacement for validation and benchmarking, ensuring that any accuracy differences introduced by sparse variants are solely due to the sparse logic, not the RoPE placement.

---

## Core Design Principle: Lazy RoPE

### The Problem with Post-RoPE Storage

In standard `FULL` / `POSTROPE` policies:

- Model layers apply RoPE to Q and K before passing them to `Attention.forward()`
- KV cache stores **post-RoPE K** (already rotated by position)
- Once K is rotated, the original amplitude/phase information is entangled with position

This means any sparse policy operating on post-RoPE KV must either:
- Work with position-baked tensors (limiting mathematical reasoning)
- Or perform lossy "inverse RoPE" to recover amplitude/phase (adding error and compute)

### PreRoPE's Solution

PreRoPE inverts this flow:

| Stage | FULL / POSTROPE | PREROPE |
|---|---|---|
| Model layer output | post-RoPE Q/K | **pre-RoPE Q/K** |
| KV cache stores | post-RoPE K | **pre-RoPE K** |
| CPU offload stores | post-RoPE K | **pre-RoPE K** |
| RoPE applied | In model, before Attention | **Lazily in policy, before kernel** |
| Attention kernel sees | post-RoPE Q/K | post-RoPE Q/K (rotated by policy) |

The critical invariant:

> From the attention kernel's perspective, PREROPE and FULL are indistinguishable. The same Q_rot, K_rot, V tensors are consumed. Only the internal storage format differs.

---

## Mathematical Foundation

### Pre-RoPE as Complex Numbers

Treat each dimension pair `(2i, 2i+1)` as a complex number:

```
z_q[i] = q_{2i} + i * q_{2i+1}
z_k[i] = k_{2i} + i * k_{2i+1}
```

Define amplitude and phase invariants that are **distance-independent**:

```
A_i = q_{2i} * k_{2i} + q_{2i+1} * k_{2i+1}
B_i = q_{2i+1} * k_{2i} - q_{2i} * k_{2i+1}
M_i = sqrt(A_i^2 + B_i^2)     # amplitude, invariant under rotation
phi_i = atan2(B_i, A_i)       # phase, invariant under rotation
```

### Post-RoPE Contribution Prediction

After RoPE rotation (Q at position t, K at position s, distance `Delta = t - s`):

```
C_i(Delta) = M_i * cos(Delta * theta_i + phi_i)
```

where `theta_i = base^(-2i/d)` is the RoPE frequency for pair i.

**Critical property**: Given pre-RoPE Q and K, the pair `(M_i, phi_i)` is fully determined. We can evaluate `C_i(Delta)` for any distance **without applying RoPE**.

This is the mathematical foundation that enables DMAS (Distance-Modulated Amplitude Scoring) and other pre-RoPE-based sparse strategies.

---

## Data Flow

### Prefill Phase

```
Model Layer          Attention.forward()          PreRoPEPolicy
    |                       |                            |
    |-- pre-RoPE q, k ---->|                            |
    |                       |-- store pre-RoPE k ------>| (KV cache / CPU offload)
    |                       |                            |
    |                       |<-- apply_rope_in_attention = True
    |                       |                            |
    |                       |-- pre-RoPE q ------------>|
    |                       |                            |
    |                       |                            |-- lazy RoPE on q (current positions)
    |                       |                            |-- lazy RoPE on k (absolute positions)
    |                       |                            |
    |                       |<-- post-RoPE q, k, v -----|
    |                       |                            |
    |                       |-- flash_attn_kernel ------>| (causal=True for current chunk)
```

For chunked prefill with CPU offload:
1. Current query tokens are rotated with their absolute positions
2. Each historical CPU block is loaded, its K is rotated with the block's absolute positions
3. Attention outputs are merged via LSE-weighted accumulation
4. Current chunk's K/V is rotated and combined with historical attention

### Decode Phase

```
PreRoPEPolicy.compute_chunked_decode()
    |
    |-- 1. Materialize prefilled CPU history (pre-RoPE K blocks)
    |-- 2. Gather accumulated decode K from decode buffer (pre-RoPE)
    |-- 3. Concatenate: [hist_pre_rope_K, decode_pre_rope_K]
    |-- 4. Apply RoPE to full K with absolute positions [0, ..., seq_len]
    |-- 5. Rotate Q with current position
    |-- 6. flash_attn_kernel(q_rot, k_rot, v)
```

Historical prefilled K blocks remain stored as pre-RoPE tensors in CPU offload. They are only rotated when needed for decode attention.

---

## Implementation Details

### `apply_rope_in_attention` Flag

`PreRoPEPolicy` sets:

```python
apply_rope_in_attention = True
```

This causes model layers (Llama, Qwen2/3, GLM-4) to pass raw pre-RoPE Q/K into `Attention.forward()`. The policy then accesses:

```python
context.positions     # absolute token positions
context.rotary_emb    # RoPE embedding table
```

and applies RoPE lazily via `_apply_rope()` / `_apply_rope_qk()`.

### Stream Synchronization (Decode Path)

The decode path exposed a subtle stream bug during validation:

**Bug pattern** (incorrect):
```python
wait_slot_layer(slot)        # on compute_stream
get_kv_for_slot(slot)        # on default stream (STALE DATA!)
```

**Fix**:
```python
with torch.cuda.stream(compute_stream):
    wait_slot_layer(slot)
    k, v = get_kv_for_slot(slot)   # same stream
    # ... process ...
torch.cuda.default_stream().wait_stream(compute_stream)
```

Any policy that waits for a slot then reads it must do both on the same stream, or explicitly bridge streams before reading.

### Block Position Reconstruction

Since pre-RoPE K blocks don't encode position, the policy must reconstruct absolute positions when applying RoPE:

```python
def _block_positions(block_index, num_tokens, block_size, device, dtype):
    start = block_index * block_size
    return torch.arange(start, start + num_tokens, device=device, dtype=dtype)
```

This ensures each historical block is rotated with its correct absolute positions in the sequence.

---

## Why Pre-RoPE Enables Novel Sparse Strategies

### Comparison with Post-RoPE

| Capability | Post-RoPE (FULL/POSTROPE) | Pre-RoPE (PREROPE) |
|---|---|---|
| KV cache stores | rotated, position-baked tensors | raw amplitude/phase |
| Extract `(M_i, phi_i)` | No (distance already baked in) | Yes (invariants are preserved) |
| Predict `C_i(Delta)` without rotation | Requires lossy inverse RoPE | Direct from pre-RoPE state |
| Subsampling before RoPE | Aliasing for high frequencies | Safe for low-frequency preservation |
| Block scoring based on amplitude/phase | Impossible | Exact, no approximation |

### Pre-RoPE as a "Source of Truth"

Pre-RoPE KV cache is the canonical representation of amplitude/phase. Any operation that needs to reason about "how much will this K contribute to attention at distance D?" can do so exactly from pre-RoPE state, without:
- Rotating vectors (O(d) per token)
- Inverting RoPE (lossy)
- Approximating with post-RoPE proxies

This enables strategies like:
- **DMAS scoring**: Predict post-RoPE block contributions without rotation
- **Distance-adaptive subsampling**: Subsample distant blocks before RoPE, preserving only low-frequency information that survives at distance
- **Sub-block granularity**: Score 256-token sub-blocks within 4096-token blocks, keeping only high-contribution regions

These strategies are implemented in the experimental `QPOOL` policy, which builds on PREROPE's pre-RoPE foundation.

---

## Validation and Equivalence

### Semantic Contract

> `PREROPE` must preserve **exactly the same generation semantics as `FULL`**, even though the cached K states are stored pre-RoPE instead of post-RoPE.

This is verified by:
1. **Unit/shape tests**: `tests/test_dense_prefill_policy.py`
2. **Behavioral alignment**: RULER runs under CPU offload + chunked prefill, exact output comparison for FULL, POSTROPE, and PREROPE

See:
- [`docs/ruler_rope_policy_alignment.md`](docs/ruler_rope_policy_alignment.md)
- [`docs/rope_policy_design.md`](docs/rope_policy_design.md)

### Density Stats

PREROPE `select_blocks` returns all available blocks (100% density), maintaining full attention semantics:

```python
def select_blocks(self, available_blocks, ...):
    return available_blocks   # density = 100%
```

The density stats reflect this:
- `total_available_blocks`: all blocks in CPU offload
- `total_selected_blocks`: same as available (100%)
- `overall_density`: 1.0

---

## Relationship to Other Policies

| Policy | KV Storage | RoPE Timing | Sparse Strategy | Use Case |
|---|---|---|---|---|
| `FULL` | post-RoPE | model layer | none (baseline) | default baseline |
| `POSTROPE` | post-RoPE | model layer | Sparge-style chunked prefill | post-RoPE sparse prefill |
| `PREROPE` | **pre-RoPE** | **lazy in policy** | none (semantic baseline) | pre-RoPE semantic baseline |
| `TRIATTENTION` | pre-RoPE | lazy in policy | decode-time token selection | decode sparse attention |
| `QPOOL` | pre-RoPE | lazy in policy | sub-block DMAS scoring | experimental prefill sparse |

- `PREROPE` vs `TRIATTENTION`: Both use pre-RoPE KV, but PREROPE is a semantic-alignment policy (should match FULL), while TRIATTENTION adds its own decode-time selection logic.
- `PREROPE` vs `QPOOL`: QPOOL extends PREROPE's pre-RoPE foundation with DMAS-based sub-block selection for sparse prefill.

---

## Files

- `nanovllm/kvcache/sparse/prerope.py` -- PreRoPEPolicy implementation
- `nanovllm/kvcache/sparse/qpool.py` -- QPoolPolicy (experimental sparse extension)
- `nanovllm/kvcache/sparse/policy.py` -- SparsePolicy base class
- `nanovllm/config.py` -- SparsePolicyType enum
- `nanovllm/models/llama.py`, `qwen2.py`, `qwen3.py`, `glm4.py` -- model layers with `apply_rope_in_attention` support
