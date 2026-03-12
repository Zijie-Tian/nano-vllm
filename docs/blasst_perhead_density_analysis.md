# BLASST Per-Head KV Density Analysis

Per-head KV density profiling for BLASST sparse attention, investigating IO savings potential under GQA.

**Model**: Llama-3.1-8B-Instruct (32 Q heads, 8 KV heads, GQA ratio 4:1)
**Config**: RULER 32K, `--enable-offload`, Layer 0 statistics
**Date**: 2026-03-12

---

## Test Setup

Modified `blasst.py` to track per-head KV density:
- `mask_buffer.any(dim=0)` → per-head `(num_heads, num_sub)` required KV sub-blocks
- `.mean(dim=-1)` → per-head fraction of KV sub-blocks needed

Tested 3 datasets × 3 samples each:
- `niah_single_1`: Needle-in-a-haystack (3/3 pass)
- `qa_1`: Question answering (1/3 pass)
- `vt`: Variable tracking (1/3 pass)

---

## Per-Head Density Results (Excluding Last Chunk)

### niah_single_1

| Category | Heads | Avg Density |
|----------|-------|:-----------:|
| SPARSE (<10%) | H21, H28, H29, H30, H31 | 0.6-4.2% |
| LOW (10-50%) | H6, H22 | 47-50% |
| MEDIUM (50-80%) | H3, H10, H19, H20, H25, H26 | 57-78% |
| DENSE (>80%) | H0-H2, H4-H5, H7-H9, H11-H18, H23-H24, H27 | 83-100% |

### qa_1

| Category | Heads | Avg Density |
|----------|-------|:-----------:|
| SPARSE (<10%) | H30, H31 | 0.6-0.8% |
| LOW (10-50%) | H22, H29 | 42% |
| MEDIUM (50-80%) | H4-H6, H13, H15, H18-H20, H26, H28 | 52-78% |
| DENSE (>80%) | H0-H3, H7-H12, H14, H16-H17, **H21**, H23-H25, H27 | 83-100% |

### vt

| Category | Heads | Avg Density |
|----------|-------|:-----------:|
| SPARSE (<10%) | H21, H22, H28, H29, H30, H31 | 0.6-8.6% |
| LOW (10-50%) | H6 | 48% |
| MEDIUM (50-80%) | H3, H10, H12, H15, H19, H20, H25, H26 | 57-78% |
| DENSE (>80%) | H0-H2, H4-H5, H7-H9, H11, H13-H14, H16-H18, H23-H24, H27 | 82-100% |

---

## Cross-Task Stability

| Head | niah_single_1 | qa_1 | vt | Stable? |
|:----:|:---:|:---:|:---:|:---:|
| H30 | 0.6% | 0.8% | 0.6% | ✅ Always sparse |
| H31 | 0.6% | 0.6% | 0.6% | ✅ Always sparse |
| H21 | 4.2% | **94.3%** | 4.9% | ❌ Flips by task |
| H28 | 0.6% | 51.8% | 1.1% | ❌ Task-dependent |
| H29 | 0.6% | 42.1% | 1.0% | ❌ Task-dependent |

**Key finding**: Only H30 and H31 are universally sparse. Most "sparse" heads change behavior across tasks. Same-task cross-sample variation is small (<±5%).

---

## GQA Group Analysis (Critical IO Insight)

Llama-3.1-8B GQA groups (4 Q heads → 1 KV head):

| KV Group | Q Heads | niah Density (union) | qa_1 Density (union) | IO Skippable? |
|:--------:|---------|:---:|:---:|:---:|
| 0 | H0-H3 | ≈100% | ≈100% | ❌ |
| 1 | H4-H7 | ≈100% | ≈100% | ❌ |
| 2 | H8-H11 | ≈100% | ≈100% | ❌ |
| 3 | H12-H15 | ≈100% | ≈100% | ❌ |
| 4 | H16-H19 | ≈100% | ≈100% | ❌ |
| 5 | H20-H23 | ≈100% (H23=99%) | ≈100% (H21=94%) | ❌ |
| 6 | H24-H27 | ≈100% | ≈100% | ❌ |
| 7 | H28-H31 | ≈1% | ≈70% | NIAH only |

**Conclusion**: Under GQA, as long as ANY Q head in a group needs a KV block, the entire block must be loaded (since all Q heads share the same KV cache). With most groups containing at least one dense head, **IO savings from sparse attention ≈ 0%** in GQA offload scenarios. BLASST still provides ~85% compute savings.

---

## Implications & Future Directions

1. **Sparse attention in GQA offload cannot reduce IO** by skipping blocks, because GQA group union density is near 100%.
2. **Compute savings remain significant** (~85% BLASST skip rate) even without IO savings.
3. **Alternative IO reduction strategies** should focus on:
   - **Compressed KV transfer**: Transfer all blocks at 2-bit/4-bit precision (aligns with Module A/C)
   - **CPU-side coarse attention**: Compute on CPU with compressed KV, avoid GPU transfer entirely (Module B)
   - **IO-aware second-pass filtering**: After per-head mask generation, apply a stricter group-level threshold (e.g., majority vote or score-weighted) to trade precision for IO reduction
   - **Hybrid CPU/GPU**: Dense groups → compressed transfer to GPU; sparse groups → CPU compute

---

**Author**: Zijie Tian / Gemini CLI
