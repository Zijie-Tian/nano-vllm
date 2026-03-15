# COMPASS Sparse Policy: Analysis & Performance Report

## Overview

COMPASS (COsine-similarity Masked Pooled Attention Sparse Selection) is a two-tier sparse attention policy for CPU-offloaded KV cache inference. It performs:

1. **CPU Tier (Coarse)**: Pooled cosine similarity + softmax + top-p selection at 128-token sub-block granularity
2. **GPU Tier (Fine)**: BLASST dynamic pruning with online softmax thresholding

The CPU tier determines which KV cache blocks to load from CPU→GPU (IO-level filtering), while the GPU tier performs per-head compute-level sparsity.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   select_blocks (CPU)                    │
│                                                         │
│  1. GPU: Q pooling (128-token groups) + GQA fold        │
│  2. Async transfer: pooled Q → pinned CPU buffer        │
│  3. CPU: Collect pre-computed pooled K                   │
│  4. CPU: Cosine similarity (bmm) → softmax → top-p     │
│  5. CPU: Build per-block sub-block mask                 │
│                                                         │
│  Output: IO block list + mask_buffer per sub-block      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│             compute_chunked_prefill (GPU)                │
│                                                         │
│  Load selected blocks via offload_engine                │
│  Apply mask_buffer to BLASST kernel                     │
│  BLASST performs per-head dynamic pruning                │
│                                                         │
│  Output: attention output                               │
└─────────────────────────────────────────────────────────┘
```

### Key Design: GPU-side Q Pooling + Async Transfer

The pooled K is pre-computed on GPU during `on_prefill_offload` (before D2H transfer). Q pooling is also done on GPU in `select_blocks`, then only the tiny pooled Q (~128KB) is transferred async to CPU via a dedicated metadata stream. This eliminates the original bottleneck of transferring the full Q tensor (32MB).

| Transfer | Shape | Size |
|----------|-------|------|
| Original Q | `[4096, 32, 128]` FP16 | **32 MB** |
| Pooled Q | `[32, 8, 128]` FP32 | **128 KB (250× smaller)** |

## Performance Profiling (128K Context, Llama-3.1-8B-Instruct)

### CPU Time Breakdown Evolution

| Step | v1 (原始) | v2 (GPU Q pool) | v3 (向量化 top-p) |
|------|-----------|-----------------|-------------------|
| cuda.sync | 3.89s (4.9%) | 4.58s (11.9%) | 4.30s (28.3%) |
| q.cpu() | **27.97s (35.3%)** | **0.15s (0.4%)** | 0.17s (1.1%) |
| Q pooling | **17.52s (22.1%)** | **0.22s (0.6%)** | 0.26s (1.7%) |
| K collect | 1.27s (1.6%) | 1.37s (3.5%) | 1.36s (8.9%) |
| cos matmul | 0.31s (0.4%) | 0.55s (1.4%) | 0.49s (3.2%) |
| top-p sel | 24.77s (31.2%) | 27.77s (72.2%) | **3.67s (24.1%)** |
| mask build | 3.62s (4.6%) | 3.84s (10.0%) | 4.98s (32.7%) |
| **Total CPU** | **79.3s** | **38.5s** | **15.2s** |

Optimizations applied:
- **v2**: GPU Q pooling + async metadata transfer via dedicated CUDA stream → eliminated q.cpu() and Q pooling
- **v3**: Vectorized top-p using batched `torch.sort` + `cumsum` + `scatter_` → eliminated Python double loop

### End-to-End Comparison (128K, 1 sample)

| Policy | select_blocks | compute_prefill | offload | **Total Prefill** | **E2E** | Accuracy |
|--------|---------------|-----------------|---------|-------------------|---------|----------|
| **FULL** | 0.00s | 89.9s | 0.21s | **89.9s** | 129.6s | ✅ 100% |
| **BLASST** (auto λ) | 0.04s | 89.7s | 0.21s | **89.9s** | 129.6s | ❌ 0% |
| **BLASST** (λ=0.0005) | 0.03s | 83.3s | 0.14s | **83.5s** | 125.2s | ✅ 100% |
| **COMPASS** (top_p=0.9) | 16.0s | 98.2s | 0.60s | **114.8s** | 167.2s | ✅ 100% |

COMPASS adds ~25s overhead vs FULL due to CPU estimation cost, while achieving 0% actual sub-block pruning.

## BLASST Lambda Sweep (128K)

| Lambda | Result | Compute Density | Prefill | E2E |
|--------|--------|-----------------|---------|-----|
| auto (~0.125) | ❌ FAIL | ~1.5% | 92.5s | 132.4s |
| 0.01 | ❌ FAIL | ~20-26% | 69.1s | 114.5s |
| 0.005 | ❌ FAIL | - | 87.6s | 133.8s |
| **0.001** | ✅ PASS | - | 92.2s | 147.9s |
| **0.0005** | ✅ PASS | - | **83.3s** | **125.2s** |
| **0.0001** | ✅ PASS | - | 101.9s | 149.1s |

Correctness threshold is between λ=0.001 and λ=0.005 for NIAH tasks at 128K.

## COMPASS top_p Sweep (128K)

| top_p | Accuracy | Sub-block Pruning | select_blocks | E2E |
|-------|----------|-------------------|---------------|-----|
| 0.9 | ✅ 100% | **0%** | 16.0s | 167s |
| 0.5 | ✅ 100% | **0%** | 15.8s | ~152s |
| 0.3 | ✅ 100% | **0%** | 15.8s | ~152s |
| 0.1 | ✅ 100% | **0%** | 15.8s | 152s |

**All top_p values yield 0% pruning.** The root cause is described below.

## Key Finding: Union Across Heads Defeats Pruning

The COMPASS per-row top-p + union strategy cannot achieve meaningful pruning because:

1. Different KV heads attend to **different positions** in the sequence
2. Each head's top-p selects a different subset of sub-blocks
3. The **union** across 8 heads × 32 Q groups = 256 rows means almost every sub-block is selected by at least one row
4. Even top_p=0.1 (very aggressive per-row) results in 0% pruning after union

```
Head 0 top-p selects: sub-blocks [1, 5, 20, 100, ...]
Head 1 top-p selects: sub-blocks [3, 8, 50, 200, ...]
Head 2 top-p selects: sub-blocks [10, 30, 80, 500, ...]
...
Union = nearly ALL sub-blocks selected → 0% pruning
```

## Two-Level Sparsity Constraint

| Level | Granularity | Module | Saves |
|-------|-------------|--------|-------|
| **IO Level** | Block × All Heads | COMPASS `select_blocks` | **PCIe bandwidth** |
| **Compute Level** | Sub-block × Per Head | BLASST `mask_buffer` | **GPU FLOPS** |

The fundamental constraint is that **IO transfers entire blocks (all heads)**. You cannot transfer only some heads of a block. Therefore IO-level filtering (COMPASS) can only skip a block when **no head** needs it — which rarely happens due to head diversity.

BLASST already handles per-head compute-level sparsity effectively (1.5% compute density at default λ).

## Future Directions

### Option A: Voting-based IO Selection (Minimal Change)
- Each head independently selects sub-blocks via top-p
- A sub-block is kept only if ≥ K heads select it (K=1 is union, K=H is intersection)
- Adjustable K controls IO filtering aggressiveness

### Option B: Per-Head IO Management (Architectural Change)
- Modify `offload_engine` to support per-head KV transfers
- Store KV cache per-head separately on CPU
- Each head independently selects and transfers its needed blocks
- Maximum IO savings but requires deep architectural changes

### Option C: Rely on GPU-side BLASST Only
- Current data shows COMPASS IO-level pruning = 0% + 16s CPU overhead = net negative
- BLASST with conservative λ (0.0005) achieves PASS with faster prefill than FULL
- Skip CPU estimation entirely, use BLASST for all sparsity

---

**Author**: Zijie Tian / Gemini CLI
**Date**: 2025-03-15
**Model**: Llama-3.1-8B-Instruct
**Hardware**: RTX 3090 24GB
