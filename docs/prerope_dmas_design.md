# PREROPE DMAS Design: Distance-Modulated Amplitude Scoring for Sparse Prefill

## Overview

This document describes **DMAS** (Distance-Modulated Amplitude Scoring), a sparse prefill design built specifically on the PREROPE policy semantics. It exploits the fact that PREROPE stores raw pre-RoPE KV cache, giving us access to the full amplitude and phase information before RoPE rotation is applied.

**Key insight**: Pre-RoPE Q/K encode per-frequency-pair amplitude `M_i` and phase `φ_i`. Combined with RoPE base frequencies `θ_i` and token distance `Δ`, we can predict post-RoPE attention contributions without ever rotating a vector.

---

## 1. Mathematical Foundation

### 1.1 Pre-RoPE as Complex Numbers

Treat each dimension pair `(2i, 2i+1)` as a complex number:

```
z_q[i] = q_{2i} + i·q_{2i+1}
z_k[i] = k_{2i} + i·k_{2i+1}
```

Define amplitude and phase invariants:

```
A_i = q_{2i}·k_{2i} + q_{2i+1}·k_{2i+1}
B_i = q_{2i+1}·k_{2i} - q_{2i}·k_{2i+1}
M_i = √(A_i² + B_i²)          # amplitude, distance-independent
φ_i = atan2(B_i, A_i)         # phase, distance-independent
```

### 1.2 Post-RoPE Contribution Prediction

After RoPE rotation (Q at position t, K at position s, distance `Δ = t - s`):

```
C_i(Δ) = M_i · cos(Δ · θ_i + φ_i)
```

where `θ_i = base^{-2i/d}` is the RoPE frequency for pair i.

**Critical property**: Given pre-RoPE Q and K, the pair `(M_i, φ_i)` is fully determined. We can evaluate `C_i(Δ)` for any distance without applying RoPE.

### 1.3 Block-Level Effective Contribution

A KV cache block contains `B` tokens (typically 4096). The average contribution of frequency pair i across the block depends on the oscillation rate:

```
avg_C_i ≈ M_i · sinc(θ_i · B / 2) · cos(Δ_mid · θ_i + φ_i)
```

where `sinc(x) = sin(x)/x`. The `sinc` term captures the intra-block oscillation cancellation:

| Frequency Type | θ_i · B | sinc(θ_i·B/2) | Effective? |
|---|---|---|---|
| High frequency (small i) | >> π | ≈ 0 | Block-level cancelled |
| Mid frequency | ~ π | < 1 | Partially attenuated |
| Low frequency (large i) | << π | ≈ 1 | Fully preserved |

For base=10000, d=128, B=4096:
- i=0 (highest): `θ_0 · B = 4096 >> π` → fully cancelled at distance
- i=63 (lowest): `θ_63 · B ≈ 0.4 < π` → fully preserved

---

## 2. Why PREROPE Uniquely Enables This

| Property | POSTROPE | PREROPE |
|---|---|---|
| KV cache stores | post-RoPE (already rotated) | pre-RoPE (raw amplitude/phase) |
| Can extract `M_i, φ_i` | No (distance already baked in) | Yes (amplitude/phase are invariants) |
| Can predict `C_i(Δ)` | Requires `invert_rope` (lossy) | Direct from pre-RoPE state |
| Adaptive subsampling | Post-RoPE subsampling = aliasing | Pre-RoPE subsampling = safe for low freq |

PREROPE's pre-RoPE KV is the "source of truth" for amplitude/phase. POSTROPE would need to first `invert_rope` (lossy) to recover these invariants, adding both error and compute overhead.

---

## 3. Three-Stage Design

### Stage 0: Block Summary Cache (`on_prefill_offload`)

Cache pre-RoPE K statistics per CPU block for fast Stage-1 scoring.

```python
def on_prefill_offload(self, cpu_block_id, layer_id, k_cache, num_valid_tokens):
    k_block = k_cache[:num_valid_tokens].float()
    d_pairs = self._head_dim // 2
    k_pairs = k_block.reshape(-1, self._num_kv_heads, d_pairs, 2)

    # Mean-pooled K pairs for coarse Q-K matching
    pooled_k = k_pairs.mean(dim=0)  # [H_kv, d_pairs, 2]

    # Per-pair norm for fast amplitude upper-bound
    pair_norms = k_pairs.norm(dim=-1).mean(dim=0)  # [H_kv, d_pairs]

    self._k_summary_cache[layer_id][cpu_block_id] = (
        pooled_k.detach(),
        pair_norms.detach(),
    )
```

**Storage cost**: `d_pairs × 2 × H_kv × 2 bytes` per block. For D=128, H_kv=8: ~1 KB per block. Negligible compared to KV cache itself.

### Stage 1: Distance-Modulated Block Selection (`select_blocks`)

Replace PREROPE's current "select all" with scored selection.

**Step 1.1**: Compute `M_i` and `φ_i` for each (Q-group, K-block, head, frequency-pair).

```python
# q_pooled: [G_q, H, d_pairs, 2] from pre-RoPE Q
# k_stack:  [B, H_kv, d_pairs, 2] from cached summaries

q0, q1 = q_pooled[..., 0], q_pooled[..., 1]  # [G, H, P]
k0, k1 = k_pooled[..., 0], k_pooled[..., 1]   # [B, H_kv, P]

A = einsum('ghp,bhp->ghbp', q0, k0) + einsum('ghp,bhp->ghbp', q1, k1)
B = einsum('ghp,bhp->ghbp', q1, k0) - einsum('ghp,bhp->ghbp', q0, k1)
M = sqrt(A^2 + B^2)           # [G, H, B, P]
phi = atan2(B, A)             # [G, H, B, P]
```

**Step 1.2**: Apply distance modulation.

```python
# sinc attenuation from intra-block oscillation
sinc_factors = sinc(thetas * block_size / (2 * pi))  # [P]

# average phase at block mid-distance
mid_distances = block_distances + block_size / 2
phase = mid_distances * thetas + phi  # [G, H, B, P]

# estimated per-pair contribution (upper bound via abs cosine)
estimated_scores = (M * sinc_factors * abs(cos(phase))).sum(dim=-1)
# [G, H, B]
```

**Step 1.3**: Aggregate across heads and select via top-p.

```python
# Fold Q heads to KV heads (GQA)
avg_scores = scores.mean(dim=1)  # [G, B]

# Normalize and top-p per Q-group
probs = softmax(avg_scores, dim=-1)
# ...top-p selection similar to POSTROPE stage-1...
```

**Result**: A subset of historical blocks is selected. Distant blocks with weak low-frequency signal are dropped.

### Stage 2: Resolution-Adaptive Compute (`compute_chunked_prefill`)

PREROPE's lazy RoPE enables **pre-RoPE subsampling** of distant blocks.

**Key theorem**: For a block at distance `D`, only frequency pairs with `θ_i < π/D` contribute significantly. These low-frequency signals vary slowly in token space, so subsampling by stride `s` (where `s < D/π`) is information-preserving.

**Adaptive stride table**:

| Distance Range | Stride | Rationale |
|---|---|---|
| D < 4K | 1 (no subsample) | All frequencies active |
| 4K ≤ D < 32K | 4 | High freq cancelled, mid freq partially active |
| D ≥ 32K | 16 | Only lowest freq preserved, safe to subsample |

**Implementation in block loop**:

```python
for block_idx, cpu_block_id in enumerate(cpu_block_table):
    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
    offload_engine.wait_slot_layer(slot)

    with torch.cuda.stream(compute_stream):
        prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
        block_distance = current_pos - (cpu_block_id + 1) * block_size

        # PREROPE-only: subsample BEFORE RoPE
        prev_k, prev_v, scale = self._adaptive_subsample(
            prev_k.squeeze(0), prev_v.squeeze(0),
            block_distance, block_size
        )

        # Positions subsampled accordingly
        prev_pos = block_positions(block_idx, len(prev_k), ...)

        # Lazy RoPE on subsampled K
        prev_k_rot = apply_rope(prev_k, prev_pos, rotary_emb).unsqueeze(0)

        # Attention: O(Q · K_stride) instead of O(Q · K_full)
        prev_o, prev_lse = flash_attn_with_lse(
            q_rot_batched, prev_k_rot, prev_v.unsqueeze(0),
            softmax_scale=softmax_scale, causal=False,
        )

        # LSE correction: subsampled tokens represent aggregate
        lse_corrected = prev_lse + log(scale)
```

**Why pre-RoPE subsampling is safe**: Post-RoPE subsampling would alias high-frequency components. Pre-RoPE subsampling preserves the low-frequency structure that actually carries information at distance.

---

## 4. Theoretical Guarantees

### 4.1 Subsampling Safety Bound

For a block at distance `D` with RoPE frequency `θ_i`, the maximum spatial frequency of the post-RoPE signal is:

```
f_max = θ_i / (2π)
```

The Nyquist criterion requires sampling rate `> 2·f_max`, i.e.:

```
stride s < 1 / (2·f_max) = π / θ_i
```

For the lowest surviving frequency at distance `D` (`θ_min ≈ π/D`):

```
s < D / π
```

Our stride table satisfies this with margin:
- D=64K: `s=16 < 64000/π ≈ 20000` ✓
- D=4K: `s=4 < 4000/π ≈ 1300` ✓

### 4.2 Score Estimation Error

The block-average contribution formula `M_i · sinc(θ_i·B/2) · cos(Δ_mid·θ_i + φ_i)` is exact to first order in `(θ_i·B)`. The second-order error is bounded by:

```
|error| ≤ M_i · (θ_i · B)^2 / 24
```

For low-frequency pairs (the ones that matter for distant blocks), `θ_i · B << 1`, so the approximation is highly accurate.

### 4.3 End-to-End Fidelity

DMAS introduces two approximations:
1. **Stage-1**: Block selection via estimated scores (may drop some blocks)
2. **Stage-2**: Intra-block subsampling (may lose fine-grained structure)

For distant blocks (>32K), both approximations are conservative:
- Dropped blocks have `M_i · sinc(...) · cos(...)` below threshold → their true contribution is small
- Subsampled blocks retain all low-frequency information → high-frequency contribution is negligible at this distance

For nearby blocks (<4K), no approximation is applied (stride=1, all blocks selected by default).

---

## 5. Integration with Existing PREROPE

### 5.1 File: `nanovllm/kvcache/sparse/prerope.py`

| Method | Change |
|---|---|
| `__init__` | Add `_k_summary_cache`, `_rope_thetas` |
| `on_prefill_offload` | New: cache pooled pre-RoPE K pairs and norms |
| `select_blocks` | Replace "return all" with DMAS scoring + top-p |
| `compute_chunked_prefill` | Add `_adaptive_subsample` call before `_apply_rope` |
| `_apply_rope` | Unchanged (still lazy RoPE) |

### 5.2 No Kernel Changes Required (Phase 1-2)

Stages 1 and 2 use existing PyTorch ops and `flash_attn_with_lse`. The adaptive subsampling is a simple tensor slice `k[::stride]` before RoPE application. No custom Triton kernel is needed for the initial implementation.

### 5.3 Optional Phase 3: Kernel-Level Frequency Pair Culling

For maximum efficiency, a custom Triton kernel could:
- Accept per-block `d_end` parameter (frequency pair cutoff)
- Skip inner-loop iterations for high-frequency pairs on distant blocks
- This is analogous to POSTROPE's `threshold_ln_lambda` skip but along the frequency dimension instead of the token dimension

---

## 6. Expected Savings

### 6.1 Stage-1 Selection Density

For a 128K context with 4096-token blocks:
- 32 historical blocks total
- Nearby blocks (<4K): always selected (~4 blocks)
- Distant blocks: estimated score may drop 30-50% depending on query
- **Expected density: 60-70%** (vs 100% for baseline PREROPE)

### 6.2 Stage-2 Subsampling Savings

| Block Distance | Stride | Compute Saved |
|---|---|---|
| < 4K | 1 | 0% |
| 4K-32K | 4 | 75% |
| > 32K | 16 | 93.75% |

For a typical 128K context with 32 blocks:
- ~4 blocks near (0% save)
- ~14 blocks mid (75% save)
- ~14 blocks far (93.75% save)
- **Overall compute: ~25% of baseline** (4x speedup for historical attention)

### 6.3 Combined Effect

Stage-1 drops 30% of blocks × Stage-2 reduces remaining by 75% = **~17.5% of original compute** for historical attention.

---

## 7. Implementation Roadmap

### Phase 1: Stage-1 Block Selection (IO Savings)
- Add `_k_summary_cache` and `on_prefill_offload`
- Implement `_compute_distance_modulated_score` in `select_blocks`
- Validate with RULER niah_single on 32K/64K context
- **Target**: Density < 80%, accuracy loss < 2%

### Phase 2: Stage-2 Adaptive Subsampling (Compute Savings)
- Add `_adaptive_subsample` and LSE correction
- Integrate into `compute_chunked_prefill` block loop
- Validate with full RULER suite
- **Target**: 3-4x historical attention speedup, overall accuracy > 90%

### Phase 3: Kernel-Level Optimization (Optional)
- Custom Triton kernel with per-block `d_end` frequency cutoff
- Merge with POSTROPE-style online skip threshold
- **Target**: 5-8x speedup with < 5% accuracy loss

---

## 8. Relationship to Other Designs

| Design | Core Mechanism | PREROPE DMAS Difference |
|---|---|---|
| POSTROPE SpargeAttn | Post-RoPE pooled QK + online skip | DMAS uses pre-RoPE amplitude/phase, no invert needed |
| XAttention BSA | Stride-based block sparse pattern | DMAS is data-dependent (query-aware), not fixed pattern |
| TriAttention | Decode-centric frequency statistics | DMAS is prefill-centric, exploits PREROPE invariants |
| TriBand (design doc) | Frequency-band KV compression at offload | DMAS compresses at compute time, not offload time |

DMAS is unique in being **query-aware** (data-dependent block selection) and **distance-adaptive** (resolution varies with token distance), both enabled by PREROPE's pre-RoPE amplitude/phase access.

---

## 9. Open Questions

1. **LSE correction for subsampled V**: Is `lse + log(stride)` sufficient, or do we need weight-based correction?
2. **GQA interaction**: Does folding Q heads to KV heads lose too much per-head selectivity?
3. **Dynamic threshold tuning**: Should `sinc` threshold or top-p be layer-dependent?
4. **Cross-batch consistency**: Does adaptive subsampling affect batched prefill determinism?

---

## 10. Summary

DMAS turns PREROPE's pre-RoPE KV cache from a "storage format choice" into a **computational advantage**. By treating each RoPE frequency pair as a complex number with invariant amplitude and phase, we can:

1. **Predict** post-RoPE attention scores without rotating vectors
2. **Select** blocks based on query-aware, distance-modulated amplitude scoring
3. **Subsample** distant blocks before RoPE application, safely preserving low-frequency information

The result is a sparse prefill policy that achieves 3-4x historical attention compute reduction while maintaining theoretical error bounds, all within the existing PREROPE lazy-RoPE framework.
