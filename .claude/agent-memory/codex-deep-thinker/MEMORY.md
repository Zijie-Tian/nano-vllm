# Codex Deep Thinker Memory

## GQA Dense Head Analysis (2026-02-08)

### Problem Pattern: GQA Union Bottleneck
- **Context**: Chunked prefill + CPU offload with Grouped Query Attention
- **Root Cause**: Multiple Q heads share KV cache → union of all heads' needs → densest head dominates
- **Observation**: In GLM-4-9B, each KV group (8 Q heads) has 1-3 dense heads forcing 80-90% KV loading
- **Impact**: Negates sparse attention benefits

### Solution Analysis Framework
For sparse attention optimization problems:
1. **Theoretical Analysis**: Derive union size formulas, best/worst/expected cases
2. **Related Work**: Survey FlexPrefill, MInference, Twilight, RetrievalAttention (2024-2025 papers)
3. **Complexity Breakdown**: Runtime overhead, memory overhead, LOC estimates
4. **Risk Assessment**: Accuracy loss modes, failure scenarios, fallback strategies

### Key Metrics for Sparse Attention
- **Density Estimation Methods**: Entropy (recommended), top-k ratio, variance
- **Overhead Calculation**: Compare estimation cost to full attention (target: <2%)
- **Speedup Formula**: Reduction / (1 - Reduction + Overhead)
- **Feasibility Threshold**: Expected benefit must exceed overhead by 10x+

### COMPASS Project Structure
- **Sparse Methods**: XAttention, FlexPrefill, MInference, AvgPool, COMPASS
- **Key Files**: `Xattn_chunked.py` (chunked prefill), `kernels.py` (Triton kernels)
- **Integration Pattern**: Add opt-in flag for new methods, maintain backward compatibility
- **Evaluation**: RULER benchmark with new metrics in `config_tasks.sh`

### Related Work Landscape (2024-2025)
- **FlexPrefill** (ICLR 2025 Oral): Query-aware patterns + cumulative attention selection
- **Twilight** (NeurIPS 2025): Hierarchical top-p pruning (post-softmax, per-head adaptive)
- **DuoAttention**: Binary classification (retrieval vs streaming heads)
- **H2O, StreamingLLM, SnapKV**: Uniform budgets, no per-head adaptivity
- **Gap**: No existing work explicitly optimizes GQA union bottleneck with pre-selection budgets

### Analysis Patterns
- **Theoretical Speedup**: Calculate union size reduction → KV transfer savings → latency improvement
- **Implementation Roadmap**: 4-phase approach (Prototype → Evaluate → Optimize → Production)
- **LOC Estimation**: ~150 per major component (density estimation, allocation, coordinator)
- **Overhead Tolerance**: 2% overhead acceptable for 40%+ benefit

## Solution 4: Q Head Regrouping (2026-02-08)

### Core Concept
- **Approach**: Inference-time reassignment of Q heads to KV groups
- **Strategy**: Concentrate dense heads into one group, keep remaining groups sparse
- **Key Advantage**: No weight modification needed, mathematically equivalent with proper indexing

### Feasibility Assessment: 7.5/10
**Strengths**:
- ✅ 40-50% KV load reduction (theoretical)
- ✅ Medium implementation complexity
- ✅ One-time profiling cost (1-3 hours)

**Risks**:
- ⚠️ Accuracy impact unknown (needs validation)
- ⚠️ Dense head stability assumption (must validate cross-dataset)
- ❌ One dense group still loads 90% KV (partial solution)

### Algorithm Design
**Optimization Objective**: `minimize max_{g} |Union(K_blocks for heads in group g)|`

**Algorithms**:
1. **Greedy Density-Based**: Sort by density, isolate top-k densest → O(H log H)
2. **Spectral Clustering**: Jaccard similarity on K block overlap → optimal grouping
3. **ILP**: Global optimum (NP-hard, for validation only)

**Profiling Metrics**:
- Density: |K_req(h)| / K_total
- Attention entropy: -Σ p_k log(p_k)
- Coverage overlap: |K_req(h_i) ∩ K_req(h_j)|
- Stability variance: Var(density across samples)

### Theoretical Analysis
**Mathematical Equivalence**: ✅ YES, if only KV group assignment changes (no Q head reordering)
```python
# Original: h → KV[h // (H/G)]
# Regrouped: h → KV[π(h)]
# Output: O @ W_o (same W_o, no reordering needed)
```

**Critical Assumption**: KV heads are interchangeable. Reality: training specializes KV heads to their Q heads → accuracy risk.

### Quantitative Estimates (GLM-4-9B, 128K)
```
Original GQA:
  - All 4 groups: ~80% KV load each
  - Total transfer: 3.2 × 603MB = 1930MB

Regrouped:
  - Group 0 (dense): 90% (12 heads)
  - Groups 1-3 (sparse): ~22% avg (20 heads)
  - Total transfer: 1.57 × 603MB = 947MB
  - Reduction: 2.03x

Speedup (limited by compute):
  - Load time: 60ms → 30ms
  - Compute: 200ms (unchanged)
  - Total: 260ms → 230ms → 1.13x speedup
```

### Implementation Strategy
**Phase 1: Profiling** (offline, one-time)
```python
# compass/src/head_regrouping.py
profile_and_regroup(model, calibration_data, output_json)
# Output: {layer_0: [0,0,0,1,1,2,2,3,3,...], ...}
```

**Phase 2: Integration** (runtime)
```python
# 3rdparty/nanovllm/nanovllm/ops/xattn.py
class XAttention:
    def __init__(self, regrouping_map=None):
        self.regrouping_map = regrouping_map

    def forward(self, Q, K, V):
        for g in range(G):
            heads_g = [h for h in range(H) if self.regrouping_map[h] == g]
            O[heads_g] = attention(Q[heads_g], K[g], V[g])
```

**No Physical Reordering**: Use runtime index mapping, preserve original weights.

### Related Work (2024)
- **CHAI** (ICML 2024): Clustered head attention (for pruning, not GQA regrouping)
- **Head Specialization** (arXiv 2510.21518): Confirms heads have consistent roles
- **GQA Paper** (arXiv 2305.13245): Fixed grouping during training, no inference optimization
- **Gap**: No prior work on inference-time GQA head regrouping (novel approach)

### Risk Mitigation
1. **Dense Head Stability**: Profile on 3+ datasets (PG19, C4, RULER) → measure variance
   - If Var(density) < 20% → stable, can generalize
   - If Var > 40% → unstable, per-task profiling needed

2. **Accuracy Validation**: RULER benchmark comparison
   - Success: accuracy drop < 2%
   - Fail: drop > 3% → abort or fine-tune

3. **Corner Cases**:
   - All heads dense → fallback to original GQA
   - Dense heads > G → allow multiple dense groups (lower benefit)

### Comparison with Other Solutions

| Solution | KV Reduction | Accuracy | Complexity | One-time Cost |
|----------|--------------|----------|------------|---------------|
| W_o Pruning | 20-30% | Medium | Low | Hours (fine-tune) |
| KV Duplication | 60-70% | **None** | Medium | None |
| Dense/Sparse Split | 50-60% | Low-Med | High | None |
| **Q Head Regrouping** | **40-50%** | **Unknown** | **Medium** | **1-3 hrs (profile)** |

**Trade-off**: Better than pruning (higher reduction), simpler than duplication (no KV copy), but accuracy risk.

### Next Steps
1. Implement greedy profiling algorithm (2-3 days)
2. Validate dense head stability (cross-dataset)
3. RULER benchmark: xattn_regrouped vs xattn baseline
4. Decision point: if accuracy OK (< 2% drop), proceed to optimization

## Solution 7: Per-Head Independent Mask (2026-02-08)

### Critical Insight: Compute vs Memory Bottleneck

**Key Discovery**: Per-head independent mask primarily reduces **computation (FLOPs)**, NOT **memory bandwidth**.

- Union-based loading already avoids redundant block transfers (each block loaded once per KV group)
- Per-head masks enable skipping computation for blocks not needed by specific heads
- **Realistic savings**: 40-75% FLOPs reduction, but only ~5-10% memory bandwidth reduction
- **Implication**: Effective in compute-bound regimes, less effective when memory bandwidth is the bottleneck

### Feasibility Assessment: 6.5/10

**Strengths**:
- ✅ 40-75% computational savings for mixed sparse/dense scenarios
- ✅ Supported by recent research (SparQ, Block Sparse Flash Attention)
- ✅ Minimal mask storage overhead (<10KB per sample)

**Risks**:
- ⚠️ Limited memory bandwidth improvement (~5-10% vs union-based)
- ⚠️ Fragmented memory access may hurt performance
- ⚠️ Complex implementation (custom kernels required)
- ❌ Load imbalance across heads (dense heads take much longer)

### Quantitative Analysis (GLM-4-9B, 7 sparse + 1 dense per group)

```
Union-based:
  FLOPs: 8 heads × 0.85 density × N blocks × ops = 6.8N ops
  Memory: 0.85 × 603MB = 513MB per group

Per-head:
  FLOPs: (7 × 0.15 + 1 × 0.85) × N = 1.9N ops
  Memory: ~0.88 × 603MB = 530MB (global union, slightly worse!)

Savings: 72% FLOPs, but -3% memory (global union > group union)
```

**Corrected Understanding**: Memory bandwidth is NOT improved because we still need to load the global union of all heads' blocks. The real benefit is skipping attention computation for blocks not needed by specific heads.

### Related Work (2024-2025)

1. **SparQ Attention** (ICML 2024):
   - Per-head sparse mask support (batch, heads, seq_len/128, seq_len/64)
   - 8× data transfer savings (but they select both Q and K elements)
   - Confirms per-head sparsity is production-ready

2. **Block Sparse Flash Attention** (2024):
   - Custom CUDA kernel with block-level sparsity (BM=128, BN=64)
   - Skips ~50% computation and memory for pruned blocks
   - Threshold-based gating with minimal overhead

3. **Triton Implementations**:
   - `hbsattn`: High-performance block sparse in Triton (close to CUDA perf)
   - `native-sparse-attention-triton`: Hardware-aligned sparse attention
   - Performance within 10-20% of hand-optimized CUDA

### Implementation Complexity

| Approach | Effort | Performance | Risk |
|----------|--------|-------------|------|
| PyTorch prototype | 1 week | Slow (for validation) | Low |
| Triton kernel | 2-3 weeks | 80-90% of CUDA | Medium |
| CUDA optimization | 4-6 weeks | Optimal | High (fragmented access) |
| **Total** | **9-13 weeks** | - | - |

### Memory Access Pattern Risks

**Primary Risk**: Fragmented access breaks coalescing
```cuda
// Problem: Each head loads different blocks
int block_idx = block_indices[head_idx][k];  // Different per head!
float kv_val = kv_blocks[block_idx * block_size + ...];  // Scattered access
```

**Mitigation Strategies**:
1. Shared memory caching: Cache loaded blocks per warp
2. Warp-level cooperation: All threads in warp load cooperatively
3. Block reordering: Sort blocks by index before loading
4. Target metric: >70% peak bandwidth utilization

### Integration with COMPASS

**Required Changes**:
1. `Xattn_chunked.py`: Modify `xattn_estimate_chunked` to return per-head masks (not per-KV-group union)
2. `kv_cache_manager.py` (new): Implement block deduplication layer
3. `block_sparse_attn_func`: Extend to support per-head masks or replace with custom kernel

**Testing Strategy**:
- Phase 1: Correctness (compare vs dense attention, max diff < 1e-3)
- Phase 2: Performance (benchmark vs union-based, target 20-30% speedup)
- Phase 3: RULER evaluation (GLM-4-9B, LLaMA-3.1-8B, Qwen2.5-7B)

### Recommendation

**⚠️ Proceed with caution**

Best for:
- ✅ Compute-bound scenarios (GPU has bandwidth, needs less compute)
- ✅ High sparsity variance (large diff between sparse/dense heads)
- ✅ Team has Triton/CUDA kernel expertise

**Alternative if memory-bound**: Consider Solution 3 (Dynamic KV Replication) or Solution 6 (Hierarchical Caching) instead.

**Validation Step**: Profile union-based approach first to confirm whether compute or memory is the bottleneck before investing in this solution.

## Links to Detailed Notes
- Solution 1 (Head-Aware Budget): `docs/gqa-dense-head-solutions/01_head_aware_budget.md`
- Solution 4 (Q Head Regrouping): `docs/gqa-dense-head-solutions/04_q_head_regrouping.md`
- Solution 7 (Per-Head Independent Mask): `docs/gqa-dense-head-solutions/07_per_head_independent_mask.md`
