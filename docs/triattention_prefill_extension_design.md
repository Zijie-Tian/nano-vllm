# TriAttention Prefill Extension: Four Design Proposals

This document records four independent design proposals for extending TriAttention's core mechanism (pre-RoPE Q/K concentration, trigonometric distance preference, and dual-signal scoring) into the **prefill phase**. These designs are intentionally independent of the existing POSTROPE, BLASST, COMPASS, or XAttention policy structures.

> **Why extend TriAttention to prefill?** TriAttention achieves 10.7x KV memory reduction and 2.5x throughput in decode by leveraging stable pre-RoPE centers to score key importance without unstable post-RoPE queries. The prefill phase, especially chunked prefill with CPU offload, has fundamentally different characteristics from decode — larger batch compute, historical KV loading from CPU, and per-query causal ranges — creating new opportunities for TriAttention-style optimization that do not exist in decode.

---

## Table of Contents

1. [Core Insight: What TriAttention Reveals](#1-core-insight-what-triattention-reveals)
2. [Scheme 1: TriHead — Per-Head Independent Historical Block Selection](#2-scheme-1-trihead--per-head-independent-historical-block-selection)
3. [Scheme 2: TriBand — Frequency-Band KV Compression at Offload Time](#3-scheme-2-triband--frequency-band-kv-compression-at-offload-time)
4. [Scheme 3: TriCausal — Per-Query Adaptive Attention Window](#4-scheme-3-tricausal--per-query-adaptive-attention-window)
5. [Scheme 4: TriCache — Distance-Aware KV Cache Layout Reorganization](#5-scheme-4-tricache--distance-aware-kv-cache-layout-reorganization)
6. [Cross-Scheme Relationships and Combination Paths](#6-cross-scheme-relationships-and-combination-paths)
7. [Implementation Priority and Validation Plan](#7-implementation-priority-and-validation-plan)

---

## 1. Core Insight: What TriAttention Reveals

### 1.1 Q/K Concentration in Pre-RoPE Space

In the pre-RoPE space, query and key vectors cluster tightly around **fixed, non-zero centers** that remain stable across all positions and input contexts. The Mean Resultant Length quantifies this:

$$R = \frac{\|\mathbb{E}[q]\|}{\mathbb{E}[\|q\|]}, \quad R_f = \frac{\|\mathbb{E}[q_f]\|}{\mathbb{E}[\|q_f\|]}$$

Empirically, ~90% of heads in Qwen3-8B exhibit $R > 0.95$ regardless of domain (Math/Coding/Chat MRL: 0.977–0.980).

### 1.2 Attention Logit as Trigonometric Series

When Q/K are concentrated around their centers, the attention logit approximates a trigonometric function of Q-K distance:

$$\text{logit}(q,k) \approx \sum_f \underbrace{\|\bar{q}_f\|\|\bar{k}_f\|}_{\text{amplitude}} \cos(\omega_f\Delta + \underbrace{\bar{\phi}_f}_{\text{phase}})$$

This means **each attention head has a characteristic distance preference curve** — determined by its center vectors' frequency energy distribution.

### 1.3 Per-Head Distance Preference Spectrum

Different heads exhibit drastically different distance preferences:

| Head Type | Dominant Frequencies | Preferred Distance | Typical Role |
|-----------|---------------------|-------------------|--------------|
| Low-frequency | Small $\omega_f$ | Long (32K+) | Long-range dependencies, global context |
| High-frequency | Large $\omega_f$ | Short (<4K) | Local patterns, syntax, n-gram-like |
| Mixed | Broad spectrum | Medium (4K–16K) | Cross-range reasoning |
| Concentrated | Single dominant peak | Specific band | Specialized semantic roles |

**This spectrum is the foundation of all four prefill extension schemes.** In decode, it is used to decide "which keys to evict." In prefill, it can be used for fundamentally different purposes: independent per-head block selection, frequency-band compression, per-query range adaptation, and physical layout reorganization.

### 1.4 Why Prefill is Different from Decode

| Characteristic | Decode Phase | Prefill Phase (Chunked) |
|---------------|-------------|------------------------|
| Query volume | 1 token per step | 4096+ tokens per chunk |
| KV access pattern | Incremental append | Bulk historical loading |
| Distance dynamics | Monotonically increasing | Fixed per chunk |
| Attention kernel | flash_attn_with_kvcache | flash_attn_varlen_func / custom Triton |
| Sparse opportunity | Token-level eviction | Block-level loading + compute skipping |
| Head independence constraint | Unified KV tensor required | Per-head subrange reading feasible |

The key insight: **prefill's bulk loading + per-head subrange feasibility creates opportunities that decode cannot exploit.**

---

## 2. Scheme 1: TriHead — Per-Head Independent Historical Block Selection

### 2.1 Core Idea

Break the constraint that "all attention heads share the same set of loaded historical KV blocks." Each head selects its own historical blocks based on its TriAttention-derived distance preference curve. Load the union, but compute per-head over only the selected subset.

### 2.2 Why This Works in Prefill (But Not Decode)

In decode, the KV cache is a persistent tensor passed to `flash_attn_with_kvcache`. All heads must see the **same KV tensor** (with the same sequence length), or the kernel interface breaks.

In chunked prefill, historical KV is loaded from CPU offload **into GPU memory per chunk**. After loading, each head can read a **subrange** of the loaded blocks without affecting other heads. The only constraint is that the union of all heads' selections must be loaded — but each head's kernel only processes its selected blocks.

### 2.3 Offline Calibration: Distance Preference Profile

For each (layer, head), compute from calibration data:

```python
# Frequency energy distribution
energy_per_band[f] = E[|q_f|] * E[|k_f|]  # [num_bands]

# Dominant distance: where the Fourier spectrum peaks
dominant_distance[head] = fourier_peak_energy(energy_per_band)

# Distance preference curve: expected attention strength vs distance
def distance_preference_curve(delta, energy_per_band, omega):
    """
    delta: Q-K distance in tokens
    Returns scalar attention strength prediction
    """
    return sum(
        energy_per_band[f] * cos(omega[f] * delta)
        for f in range(num_bands)
    )

# Effective attention range: distance beyond which contribution < threshold
effective_range[head] = max_distance_where(
    distance_preference_curve(delta) > RANGE_THRESHOLD
)
```

### 2.4 Runtime Block Selection

For a current chunk starting at position `P_chunk`, with historical blocks `B_0, B_1, ..., B_{N-1}` (each `block_size` tokens, block `B_i` spans `[i*block_size, (i+1)*block_size)`):

```python
# Per-head block scoring
for head h in all_heads:
    for block_idx, block in enumerate(historical_blocks):
        block_mid = block_idx * block_size + block_size // 2
        delta = P_chunk - block_mid  # typical distance to this block
        relevance[h, block_idx] = distance_preference_curve_h(delta)

    # Select top-k blocks for this head
    # k_h depends on head concentration: high R -> strict top-k, low R -> conservative
    k_h = compute_head_budget(h, total_budget, concentration_R[h])
    selected_blocks[h] = top_k_indices(relevance[h], k=k_h)

# Union for loading
blocks_to_load = union_over_heads(selected_blocks)
# blocks_to_load.size <= sum(k_h), but typically much smaller than N
# due to overlap: short-range and long-range heads both need nearby blocks
```

### 2.5 Compute: Per-Head Masked Attention Kernel

The loaded blocks (union) are placed in GPU memory contiguously. Each head's kernel only reads the blocks it selected:

```python
# Kernel pseudocode (Triton-style)
for each head h:
    for each query token q in chunk:
        m_global = -inf
        acc = 0
        lse = 0
        for each block B where mask[h, B] = 1:
            load K_B, V_B into SRAM
            qk = Q[q] @ K_B^T * softmax_scale
            apply causal mask
            m_local = max(qk, axis=1)
            # Standard online softmax
            m_new = max(m_global, m_local)
            scale = exp(m_global - m_new)
            p = exp(qk - m_new)
            acc = acc * scale + p @ V_B
            lse = lse * scale + sum(p)
            m_global = m_new
        output[h, q] = acc / lse
```

**Key difference from standard chunked prefill:** The inner loop over blocks is gated by a per-head boolean mask. Heads with short effective ranges skip distant blocks entirely, saving compute and memory bandwidth.

### 2.6 GQA Handling

With Grouped Query Attention (GQA), multiple query heads share the same KV head. The selection must be at KV-head granularity:

```python
# Group query heads by KV head
for kv_head in range(num_kv_heads):
    query_heads_in_group = range(
        kv_head * num_key_value_groups,
        (kv_head + 1) * num_key_value_groups
    )
    # Union of selected blocks within the group
    group_selected = union(
        selected_blocks[h] for h in query_heads_in_group
    )
    # All query heads in this group use group_selected
```

### 2.7 Expected Benefits and Risks

| Aspect | Benefit | Risk |
|--------|---------|------|
| **Sparsity** | Per-head selection captures true distance needs; short-range heads avoid loading distant blocks | Union may still be large if many heads are broad-spectrum |
| **Compute** | Skipped blocks = zero compute for those heads | Kernel complexity increases with per-head mask |
| **Communication** | Union size < total blocks if selections diverge | Overhead of per-head mask storage |
| **Accuracy** | No approximation: exact attention over selected subset | Missed blocks cannot be recovered without re-computation |
| **Memory** | Reduced peak KV memory on GPU | Mask storage: [num_heads, num_blocks] booleans |

### 2.8 Quantitative Estimate

For a 128K context with 4096 block size: 32 historical blocks.

| Scenario | Union Load | Density (vs Full) |
|----------|-----------|-------------------|
| All heads need all blocks | 32 | 100% |
| Mixed head spectrum (50% short, 30% medium, 20% long) | ~18–22 | 56–69% |
| Mostly short-range heads | ~10–14 | 31–44% |
| Full TriHead + TriCausal combined | ~8–12 | 25–38% |

---

## 3. Scheme 2: TriBand — Frequency-Band KV Compression at Offload Time

### 3.1 Core Idea

Instead of discarding entire KV tokens (token-level sparsity), compress KV at the **frequency-band level**. Different frequency bands within a single token have different importance for future attention. Low-frequency bands (long-range) remain important at distance; high-frequency bands (short-range) decay rapidly and can be aggressively compressed.

### 3.2 Mathematical Foundation

RoPE splits the head dimension into d/2 frequency pairs. Each pair corresponds to a rotation frequency $\omega_f = \theta^{-2f/d}$. The attention contribution of band f at distance $\Delta$ is:

$$\text{contrib}_f(\Delta) = \|q_f\| \cdot \|k_f\| \cdot \cos(\omega_f \Delta + \phi_f)$$

For **fixed centers** (TriAttention assumption):

$$\mathbb{E}[\text{contrib}_f(\Delta)] \approx \|\bar{q}_f\| \cdot \|\bar{k}_f\| \cdot \cos(\omega_f \Delta + \bar{\phi}_f)$$

The envelope decays as $\cos(\omega_f \Delta)$ oscillates. Critically:
- **Low $\omega_f$** (small f): $\cos(\omega_f \Delta)$ oscillates slowly → maintains magnitude over long distances
- **High $\omega_f$** (large f): $\cos(\omega_f \Delta)$ oscillates rapidly → averages to near-zero over distance windows

### 3.3 Per-Band Importance as Function of Future Distance

For a KV token at position `pos_kv`, define its future expected usage:

```python
# Expected attention contribution from this KV token
# Averaged over all future query positions [pos_kv + 1, pos_kv + expected_window]

for band f in range(num_bands):
    # Average contribution over future distances
    avg_contrib[f] = mean(
        energy_f * cos(omega[f] * delta)
        for delta in range(1, expected_window + 1)
    )

# High-frequency bands: avg_contrib ≈ 0 (rapid oscillation cancels out)
# Low-frequency bands: avg_contrib > 0 (slow oscillation, sustained contribution)
```

### 3.4 Compression Strategy by Band Importance

```python
def compress_kv_token(kv_token, position, expected_window, energy_profile):
    """
    kv_token: [head_dim] vector (pre-RoPE or post-RoPE)
    Returns compressed representation
    """
    k_complex = to_complex_pairs(kv_token)  # [num_bands] complex
    compressed = []

    for f in range(num_bands):
        importance = band_importance(f, position, expected_window, energy_profile)

        if importance > HIGH_THRESHOLD:
            # Band is important: retain full precision
            compressed.append((f, k_complex[f], precision='fp16'))
        elif importance > MEDIUM_THRESHOLD:
            # Moderate importance: reduce precision
            compressed.append((f, quantize(k_complex[f], bits=8), precision='int8'))
        elif importance > LOW_THRESHOLD:
            # Low importance: heavy quantization
            compressed.append((f, quantize(k_complex[f], bits=4), precision='int4'))
        else:
            # Negligible: store mean or zero
            compressed.append((f, band_mean[f], precision='mean'))

    return compressed
```

### 3.5 Distance-Layered Compression

Instead of per-token compression, apply **distance-dependent compression** at the block level:

```python
# For historical blocks at different distances from current chunk:
distance_layers = [
    (0, 4096,    'fp16'),      # Nearest: full precision
    (4096, 16384, 'int8'),     # Near: 2x compression
    (16384, 65536, 'int4'),    # Medium: 4x compression
    (65536, inf,   'mean'),    # Far: store per-band mean only
]

# At offload time, based on block's distance from current position:
for block in historical_blocks:
    distance = current_pos - block.end_pos
    for layer_start, layer_end, precision in distance_layers:
        if layer_start <= distance < layer_end:
            compress_block(block, precision)
            break
```

### 3.6 Decode Compatibility

TriBand compression must remain compatible with decode-phase TriAttention:
- Decode TriAttention operates on **pre-RoPE** keys (for scoring)
- If KV is stored compressed, it must be decompressible to pre-RoPE form for scoring
- Post-RoPE KV (POSTROPE policy) cannot be directly used for TriAttention scoring

**Resolution:** Store both:
1. Compressed post-RoPE KV for attention compute
2. Per-band summary statistics (from pre-RoPE analysis) for TriAttention scoring

Or use **PREROPE policy** (pre-RoPE KV storage), where KV is stored in pre-RoPE form and TriBand compression applies directly.

### 3.7 Expected Benefits and Risks

| Aspect | Benefit | Risk |
|--------|---------|------|
| **Granularity** | Finer than token-level: a token's low-freq bands may survive while high-freq are compressed | More complex storage format |
| **Accuracy** | Low-freq bands (most important for long-range) preserved | High-freq band loss may affect local pattern recognition |
| **Memory** | 2–4x KV memory reduction depending on distance | Decompression overhead at load time |
| **Compatibility** | Can integrate with existing block-based offload | Requires PREROPE or dual storage |

---

## 4. Scheme 3: TriCausal — Per-Query Adaptive Attention Window

### 4.1 Core Idea

Within a single prefill chunk, different query tokens have different distances to historical KV. A query at the **beginning** of the chunk is closer to historical KV than a query at the **end**. Using per-head distance preferences, each query token can have its own **adaptive historical window** — loading fewer blocks for early queries in the chunk.

### 4.2 The Causal Distance Spectrum in a Chunk

For a chunk spanning `[P, P + L)` and historical KV spanning `[0, P)`:

| Query Position | Distance to Nearest Historical KV | Distance to Farthest Historical KV |
|---------------|-----------------------------------|-----------------------------------|
| `q_0` at `P` | 1 token | `P` tokens |
| `q_{L/2}` at `P + L/2` | `L/2 + 1` tokens | `P + L/2` tokens |
| `q_{L-1}` at `P + L - 1` | `L` tokens | `P + L - 1` tokens |

For a head with effective range `R_eff`:
- `q_0`: Historical KV from `[P - R_eff, P)` is relevant
- `q_{L-1}`: Historical KV from `[P + L - 1 - R_eff, P)` is relevant

**The first query in the chunk needs fewer historical blocks than the last query.**

### 4.3 Per-Query Block Window Calculation

```python
def adaptive_window_for_query(query_pos, head, block_size, num_historical_blocks):
    """
    Returns the range of historical block indices this query needs.
    """
    r_eff = effective_range[head]  # From TriAttention calibration

    # Relevant historical positions: [query_pos - r_eff, query_pos)
    # But historical KV only exists up to current chunk start
    hist_start = max(0, query_pos - r_eff)
    hist_end = query_pos  # exclusive

    # Convert to block indices
    first_block = hist_start // block_size
    last_block = (hist_end - 1) // block_size

    return range(first_block, last_block + 1)
```

### 4.4 Query Grouping for Efficient Kernel Execution

Running a separate kernel per query is impractical. Instead, group queries by their "historical block needs":

```python
# For each head, bucket queries by their first required historical block
# (which determines how many blocks they need)

for head in all_heads:
    query_groups = defaultdict(list)
    for q_idx, q_pos in enumerate(chunk_positions):
        first_block = max(0, q_pos - effective_range[head]) // block_size
        query_groups[first_block].append(q_idx)

# Sort groups: group with largest first_block (fewest historical blocks) first
sorted_groups = sorted(query_groups.items(), key=lambda x: -x[0])

# Execute: each group gets a kernel launch with its specific block range
```

### 4.5 Progressive Loading Pipeline

```python
# As we move through query groups in a chunk:
# Group 0 (early queries, large first_block): needs blocks [k, N)
# Group 1 (mid queries, medium first_block): needs blocks [m, N), where m < k
# Group 2 (late queries, small first_block): needs blocks [0, N)

# Loading strategy:
# 1. Start with Group 0: load blocks [k, N)
# 2. For Group 1: additionally load blocks [m, k)
# 3. For Group 2: additionally load blocks [0, m)

# This is naturally compatible with ring buffer pipeline:
# as we progress through groups, we load additional blocks into new slots
```

### 4.6 Combining with TriHead

TriCausal and TriHead are orthogonal and highly complementary:

| Dimension | TriHead Decides | TriCausal Decides |
|-----------|----------------|-------------------|
| Which blocks? | Per-head distance preference | Per-query adaptive range |
| How many blocks? | Fixed per-head budget | Variable per-query group |
| When to load? | At chunk start | Progressive through chunk |

Combined:
```python
# For each head h and query group g:
#   candidate_blocks = selected_blocks[h]  # from TriHead
#   adaptive_range = adaptive_window_for_query(group_queries, h)  # from TriCausal
#   final_blocks = intersection(candidate_blocks, adaptive_range)
```

### 4.7 Expected Benefits and Risks

| Aspect | Benefit | Risk |
|--------|---------|------|
| **Early queries** | May need only 20–40% of historical blocks | Kernel launch overhead from multiple groups |
| **Late queries** | Still get full range; no accuracy loss | Group scheduling complexity |
| **Compute** | Early queries significantly cheaper | Group boundary effects |
| **Pipeline** | Natural fit for ring buffer progressive loading | Requires careful stream synchronization |

---

## 5. Scheme 4: TriCache — Distance-Aware KV Cache Layout Reorganization

### 5.1 Core Idea

Reorganize the physical storage layout of KV cache so that tokens with similar "distance relevance" are stored contiguously. This enables efficient bulk loading of "distance layers" and supports variable compression rates per layer.

### 5.2 The Problem with Time-Ordered Storage

Current KV cache storage: `[token_0, token_1, token_2, ..., token_N]` in temporal order.

For a query at position `P`, the relevant tokens form a **contiguous suffix** `[P - R_eff, P)`. But:
- Different heads have different `R_eff`
- Loading the full suffix wastes bandwidth for short-range heads
- Loading per-head subranges causes non-contiguous reads

### 5.3 Distance-Layered Layout

Instead of time-order, organize by **distance from current position**:

```python
# At time T (current position), reorganize historical KV into layers:
# Layer 0 (Immediate): [T-512, T)      — all heads need this
# Layer 1 (Near):      [T-2048, T-512) — most heads need this
# Layer 2 (Mid):       [T-8192, T-2048)
# Layer 3 (Far):       [T-32768, T-8192)
# Layer 4 (Distant):   [0, T-32768)     — only long-range heads

# Physical layout: [Layer0 | Layer1 | Layer2 | Layer3 | Layer4]
# Within each layer: time-ordered for causal correctness
```

### 5.4 Layer Properties Table

| Layer | Distance Range | Head Coverage | Compression | Load Priority |
|-------|---------------|---------------|-------------|---------------|
| 0 | [0, 512) | 100% (all heads) | None (fp16) | Always |
| 1 | [512, 2048) | ~95% | Light (int8) | Always |
| 2 | [2048, 8192) | ~70% | Medium (int4) | Conditional |
| 3 | [8192, 32768) | ~40% | Heavy (int4 + sparse) | On-demand |
| 4 | [32768, inf) | ~15% | Extreme (top-k only) | Rarely |

### 5.5 Incremental Reorganization During Prefill

Reorganization does not need to happen all at once:

```python
# As each new chunk is prefilled:
# 1. The new chunk becomes the new "current" reference point
# 2. Historical tokens shift into different distance layers
# 3. Only tokens that cross layer boundaries need physical movement

# Example: chunk size = 4096
# Before: current = 32000, layers based on 32000
# After:  current = 36096, layers based on 36096
# Tokens in [32000, 36096) move from "current" to "Layer 0"
# Tokens that were in Layer 0 may shift to Layer 1, etc.

# Physical movement is batched: only move when a full block crosses boundary
```

### 5.6 Layer-Selective Loading

```python
# For each head, determine which layers it needs:
for head in all_heads:
    r_eff = effective_range[head]
    needed_layers = [layer for layer in layers if layer.min_dist < r_eff]
    # Load only needed_layers
    # Within each loaded layer, read full layer (contiguous!)
```

### 5.7 Integration with Block-Based Offload

The existing system uses 4096-token blocks for CPU offload. TriCache can work within this constraint:

```python
# Each 4096 block is internally partitioned by distance layer
# Block metadata stores: "this block contributes to layers [L_i, L_j]"
# At load time, only read the relevant layer partitions from each block

# Or: assign entire blocks to layers
# Blocks closest to current position -> Layer 0
# Middle blocks -> Layer 1, 2
# Farthest blocks -> Layer 3, 4
```

### 5.8 Expected Benefits and Risks

| Aspect | Benefit | Risk |
|--------|---------|------|
| **Contiguity** | Same-layer tokens are contiguous → efficient DMA | Reorganization overhead |
| **Loading** | Per-head layer selection → minimal waste | Layer boundary management |
| **Compression** | Different compression per layer | Complex decompression pipeline |
| **Memory** | Far layers heavily compressed → extends max context | Reorganization memory churn |

---

## 6. Cross-Scheme Relationships and Combination Paths

### 6.1 Orthogonality Matrix

| | TriHead | TriBand | TriCausal | TriCache |
|---|:---:|:---:|:---:|:---:|
| **TriHead** | — | Partial | Full | Full |
| **TriBand** | Partial | — | Independent | Full |
| **TriCausal** | Full | Independent | — | Full |
| **TriCache** | Full | Full | Full | — |

- **Full orthogonal**: Can be combined without conceptual conflict
- **Partial overlap**: Shared mechanism but different application point
- **Independent**: Addresses completely different bottleneck

### 6.2 Recommended Combination Stacks

**Stack A: Minimal Viable — TriHead Only**
- Change: Replace 1D `selected_blocks` with per-head selection
- Benefit: Head-level sparsity without storage changes
- Complexity: Medium

**Stack B: Compute Optimization — TriHead + TriCausal**
- TriHead: which blocks each head needs
- TriCausal: which queries need which blocks
- Combined: per-head-per-query block intersection
- Benefit: Maximum compute reduction
- Complexity: High (kernel changes)

**Stack C: Memory Optimization — TriBand + TriCache**
- TriCache: distance-layered physical layout
- TriBand: frequency-band compression within layers
- Combined: 4–8x KV memory reduction with efficient loading
- Benefit: Extends max context length
- Complexity: High (storage format changes)

**Stack D: Full Stack — All Four**
- TriCache: organize storage by distance layer
- TriBand: compress within layers by frequency
- TriHead: per-head select from loaded layers
- TriCausal: per-query adapt within selected blocks
- Benefit: Maximum sparsity + compression
- Complexity: Very high

### 6.3 Implementation Dependency Graph

```
TriCache (storage layer)
    |
    v
TriBand (compression within layers)
    |
    v
TriHead (selection from loaded blocks)
    |
    v
TriCausal (query-level adaptation)
```

Each layer depends on the layer below but can be implemented independently.

---

## 7. Implementation Priority and Validation Plan

### 7.1 Phase 1: TriHead Prototype

**Goal:** Prove per-head block selection reduces loaded blocks without accuracy loss.

**Steps:**
1. Extend `on_prefill_offload` to cache per-head distance preference data
2. Modify `select_blocks` to return per-head block lists (instead of 1D list)
3. Modify `compute_chunked_prefill` to accept per-head masks
4. Implement simple union-based loading + per-head masked kernel
5. Validate on RULER niah_single_1 single sample

**Validation Criteria:**
- Accuracy: match FULL baseline on needle test
- Density: loaded blocks < 70% of available for 32K+ context
- Overhead: per-head mask storage < 1% of KV memory

### 7.2 Phase 2: TriCausal Integration

**Goal:** Reduce compute for early queries in a chunk.

**Steps:**
1. Group chunk queries by adaptive window
2. Launch multiple kernel groups with different block ranges
3. Measure compute time reduction for chunk front vs chunk back

**Validation Criteria:**
- Early query groups use < 50% of historical blocks
- End-to-end prefill time reduction: > 15%
- No accuracy regression

### 7.3 Phase 3: TriBand + TriCache

**Goal:** Reduce KV memory footprint and extend max context.

**Steps:**
1. Implement frequency-band analysis in calibration script
2. Modify offload storage format to support distance layers
3. Implement layer-selective loading in `OffloadEngine`
4. Implement band-level decompression kernel

**Validation Criteria:**
- KV memory reduction: > 3x for 128K context
- Load time reduction: > 20% (due to compression)
- Decompression overhead: < 5% of compute time

### 7.4 Success Metrics

| Metric | Baseline (POSTROPE) | TriHead Target | Full Stack Target |
|--------|--------------------|----------------|-------------------|
| Stage-1 density | 92% | 60–70% | 40–50% |
| H2D communication | 100% | 60–70% | 30–40% |
| Peak KV GPU memory | 100% | 100% | 25–30% |
| Prefill time (32K) | 100% | 80–85% | 60–70% |
| RULER accuracy | 5/5 | 5/5 | 5/5 |

---

## References

### Internal
- [`docs/rope_policy_design.md`](rope_policy_design.md) — PREROPE/POSTROPE semantic split
- [`docs/postrope_sparge_chunked_prefill_design.md`](postrope_sparge_chunked_prefill_design.md) — Current POSTROPE two-stage design
- `nanovllm/kvcache/sparse/triattention.py` — Existing TriAttention policy (decode)
- `nanovllm/kvcache/sparse/triattention_decode_selector.py` — Decode selector implementation

### External
- TriAttention paper: https://arxiv.org/abs/2604.04921
- Official repository: https://github.com/WeianMao/triattention
