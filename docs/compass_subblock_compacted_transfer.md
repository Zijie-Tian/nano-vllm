# COMPASS Sub-Block Compacted Transfer

## Overview

COMPASS (COsine-similarity Masked Pooled Attention Sparse Selection) 实现了两级稀疏注意力：CPU 端粗筛 + GPU 端细筛。本文档描述其核心特性——**Sub-Block Compacted Transfer（子块聚合传输）**，以及两级剪枝的工作原理和实验结果。

## 问题背景

原始 OffloadEngine 的 IO 粒度为整个 chunk（4096 tokens）。这对 COMPASS 造成两个问题：

1. **IO 浪费**：一个 chunk 中只要有 1 个 128-token sub-block 被选中，整个 4096-token chunk 都要传输（32× 冗余）
2. **低利用率**：若按单个 128-token sub-block 传输，PCIe 带宽利用率极低（小包传输不饱和）

## 解决方案：Gather + Bulk Transfer

```
CPU Memory (Pinned)                    Staging Buffer (Pinned)         GPU Slot
┌─────────────────────┐                ┌─────────────────┐            ┌─────────────┐
│ Block 0             │                │                 │    bulk    │             │
│  [sub0] ← selected ─┼── gather ──→  │ [sub0_blk0]     │ ═══H2D═══>│ [sub0_blk0] │
│  [sub1]             │                │ [sub2_blk0]     │            │ [sub2_blk0] │
│  [sub2] ← selected ─┼── gather ──→  │ [sub5_blk1]     │            │ [sub5_blk1] │
│  ...                │                │ [sub6_blk1]     │            │ [sub6_blk1] │
│  [sub31]            │                │ ...             │            │ ...         │
├─────────────────────┤                └─────────────────┘            └─────────────┘
│ Block 1             │                   ↑                             ↑
│  ...                │              CPU gather               Single async DMA
│  [sub5] ← selected ─┼── gather ──→  (sequential copy)       (PCIe saturated)
│  [sub6] ← selected ─┼── gather ──→
│  ...                │
└─────────────────────┘
```

### 核心 API

| 方法 | 位置 | 功能 |
|------|------|------|
| `load_staging_to_slot()` | `offload_engine.py` | CPU→GPU: staging buffer 整体异步 H2D 到 GPU slot |
| `SubBlockSelection` | `policy.py` | 数据结构: `entries = [(block_id, [sub_idx...])]` |

> **Note**: `gather_subblocks_to_staging()` 已在死代码清理中移除，staging gather 逻辑现在内联于 `compass.py` 的 `compute_chunked_prefill` 中。

### Pipeline 设计

```
Timeline ──────────────────────────────────────►

Batch 0:  [gather₀] [H2D₀] [compute₀]
Batch 1:            [gather₁] [H2D₁] [compute₁]
Batch 2:                      [gather₂] [H2D₂] [compute₂]
                                         ...

Constraint: staging buffer 共享 → 同一时刻最多 1 个 gather + 1 个 H2D
```

> **Bug Fix**: 初版尝试同时预加载多个 batch 到共享 staging buffer，导致数据竞争（gather 覆盖未完成 H2D 的数据）。修复为仅预加载 1 个 batch，pipeline 中每次只提前 1 步。

---

## 两级剪枝架构

```
                    ┌─────────────────────────────┐
                    │  L1: CPU 粗筛 (COMPASS)      │
                    │                             │
                    │  scores = Q_pool × K_pool   │
                    │  avg = mean(scores, dim=Q)   │  ← 修复: 先聚合 Q 再 top-p
                    │  probs = softmax(avg)        │
                    │  mask = top_p(probs)          │
                    │  union across heads           │
                    │                             │
                    │  → 选中的 sub-blocks          │
                    │  → gather + bulk H2D          │
                    └──────────┬──────────────────┘
                               │ 仅传输选中数据
                               ▼
                    ┌─────────────────────────────┐
                    │  L2: GPU 细筛 (BLASST)       │
                    │                             │
                    │  动态阈值 λ                   │
                    │  per-head softmax pruning     │
                    │  skip low-attention blocks    │
                    │                             │
                    │  → 最终 attention output      │
                    └─────────────────────────────┘
```

### L1 top-p 选择修复

**原始实现 (Bug)**：对 `[H, G_q, G_k]` 每行独立 top-p → union 128 个投票者 → ~100% coverage

```python
# 错误: 128 个独立 top-p 集合的并集覆盖一切
probs = softmax(scores, dim=-1)        # [H, G_q, G_k]
mask = top_p(probs)                     # per-row
overall = mask.any(dim=0).any(dim=0)    # union → ~100%
```

**修复后**：先 mean-aggregate Q → `[H, G_k]` → softmax → top-p → union across H only

```python
# 正确: 先聚合 Q 维度, 减少投票者至 H 个
avg_scores = scores.mean(dim=1)        # [H, G_k]
probs = softmax(avg_scores, dim=-1)    # [H, G_k]
mask = top_p(probs)                    # per-head
overall = mask.any(dim=0)              # union across H only
```

---

## 实验结果

### GLM-4-9B-Chat-1M, 32K Context, NIAH

| top_p | L1 CPU Pruning | L2 BLASST Pruning | Overall Pruning | Compute Density | Accuracy |
|-------|---------------|-------------------|-----------------|-----------------|----------|
| 0.9   | 0.1%          | —                 | 0.1%            | ~1.0            | ✅ 100%  |
| **0.5** | **31.3%** | **14.4%** | **41.1%** | **0.589** | ✅ 100% |
| **0.3** | **50.2%** | **11.3%** | **55.8%** | **0.442** | ✅ 100% |

- L1 pruning 控制 IO 量（PCIe 带宽节省）
- L2 pruning 进一步减少 GPU 计算量（FLOPS 节省）
- BLASST L2 在不同 top_p 下稳定 ~11-14%

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `nanovllm/config.py` | 新增 `compass_top_p` 字段 |
| `nanovllm/kvcache/offload_engine.py` | staging buffer + `load_staging_to_slot()` (注: `gather_subblocks_to_staging()` 已移除) |
| `nanovllm/kvcache/sparse/policy.py` | 新增 `SubBlockSelection` dataclass |
| `nanovllm/kvcache/sparse/compass.py` | 重写 `compute_chunked_prefill`（gather 管线）; 修复 top-p; 新增 BLASST L2 stats |
| `tests/test_ruler.py` | 修复 `compass_top_p` 参数传递; 新增 L2 统计输出 |
| `tests/test_gather_attention.py` | 新增 gather + packed attention 正确性测试 |

---

**Author**: Zijie Tian / Gemini CLI
**Date**: 2025-03-16
**Hardware**: RTX 3090 24GB
