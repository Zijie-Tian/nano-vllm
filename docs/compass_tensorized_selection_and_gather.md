# COMPASS Tensorized Selection & Vectorized Gather

本文档记录 COMPASS `select_blocks` 和 `compute_chunked_prefill` 中两阶段 CPU 侧性能优化的设计思路、实现细节与验证结果。

---

## 1. 问题背景

COMPASS 的 CPU 侧流水线包含两个关键步骤：

1. **`select_blocks` → `compass_build_entries`**：将 top-p 筛选产生的 2D boolean mask 转换为 per-head 的 sub-block 选择信息。
2. **`compute_chunked_prefill` → `V2 Packed Gather`**：根据选择信息，将 CPU KV cache 中选中的 sub-block 复制到 staging buffer，准备 H2D DMA 传输。

nsys 分析发现两者均为 CPU 端瓶颈，原因是大量 Python-level 数据结构操作和细粒度 `copy_()` 调用。

---

## 2. 优化一：TensorSelection 替代 PerHeadSubBlockSelection

### 2.1 原始问题

`compass_build_entries` 使用 O(G_k × H) 的 Python 双重循环，将 2D boolean mask 按 head 分组到 `dict[int, list[int]]` 中：

```python
# 原始：12.7ms/call，占 select_blocks 的 ~40%
for gk in range(G_k):          # ~224 sub-blocks
    bid, si = subblock_to_block[gk]
    for h in range(H):         # 8 heads
        if selected_mask[h, gk]:
            per_head_grouped[h].setdefault(bid, []).append(si)
```

### 2.2 设计方案

引入 `TensorSelection` dataclass（`policy.py`），用预分配的 3D bool tensor 替代 Python dict/list：

```
TensorSelection:
  mask:      [H_kv, max_blocks, subs_per_block]  bool, CPU
  block_ids: [max_blocks]                         int32, CPU
```

关键：`selected_mask[H, G_k]` 本身已包含 per-head per-sub-block 的选择信息。只需将其 **reshape** 为 `[H, num_blocks, subs_per_block]` 即可直接得到 per-head per-block 的 sub-block mask，无需任何遍历：

```python
# 优化后：O(1) tensor view + copy，0.05ms/call
mask_3d = selected_mask.view(H, num_blocks, subs_per_block)
self._sel_mask[:H, :num_blocks, :subs_per_block].copy_(mask_3d)
```

### 2.3 下游适配

| 位置 | 变更 |
|------|------|
| `compass.py: alloc_policy_metadata` | 预分配 `_sel_mask` 和 `_sel_block_ids` tensors |
| `compass.py: select_blocks` | 用 `view()` + `copy_()` 替代 Python 循环 |
| `compass.py: compute_chunked_prefill` | `mask.any(dim=(0,2))` + `nonzero()` 替代 Python set 操作 |
| `offload_engine.py` | 新增 `gather_packed_from_mask()` 接受 tensor mask |
| 日志/stats | 用 `mask.sum()`, `any()` 等 tensor ops 替代 Python 计数 |

### 2.4 性能结果

| 指标 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| `compass_build_entries` 单次 | 12.7ms | 0.05ms | **254×** |
| 占 `select_blocks` 比例 | ~40% | <1% | — |

---

## 3. 优化二：Vectorized Fancy + Boolean Indexing Gather

### 3.1 原始问题

`gather_packed_from_mask` 使用三层嵌套循环处理 CPU memcpy：

```
for h in range(H_kv):              # 8 heads
  for b_idx in range(num_active):  # 2-3 blocks
    nonzero() → tolist() → while coalesce runs:
      copy_() K, copy_() V         # 3-5 runs × 2
```

每个 pipeline chunk 执行 **320-640 次 `copy_()`**，大量 Python→C++ 调用开销。

### 3.2 设计方案

两步向量化：

**Step 1：Pre-expand mask 到 token 级别**（单次 tensor op）

```python
# [H_kv, num_active, subs_per_block] → [H_kv, num_active, block_size]
token_mask = mask.repeat_interleave(sub_block_size, dim=2)
```

**Step 2：Pre-compute kv_indptr**（单次 tensor op）

```python
per_head_tokens = token_mask.sum(dim=(1, 2))   # [H_kv]
kv_indptr = torch.zeros(H_kv + 1, dtype=torch.int32)
kv_indptr[1:] = per_head_tokens.cumsum(0).to(torch.int32)
```

**Step 3：单层 H_kv 循环 + fancy indexing + 2D boolean indexing**

```python
bid_list = block_ids.tolist()  # 转换一次

for h in range(H_kv):
    # Fancy index: 一次 gather 所有 active blocks
    k_blocks = k_cache_cpu[layer, bid_list, h]  # [num_active, block_size, dim]
    v_blocks = v_cache_cpu[layer, bid_list, h]

    # 2D boolean index: 跨所有 blocks 提取选中 tokens
    head_mask = token_mask[h]                    # [num_active, block_size]
    k_staging[off:off+n] = k_blocks[head_mask]   # [n_selected, dim]
    v_staging[off:off+n] = v_blocks[head_mask]
```

### 3.3 关键设计决策

- **无法消除 `for h` 循环**：staging buffer 要求 per-head 连续排列（`kv_indptr` 标记 head 边界），不同 head 写入不同区域。
- **内层 `b_idx` 循环完全消除**：fancy indexing `cpu[layer, bid_list, h]` 一次获取所有 blocks 的数据为 `[num_active, block_size, dim]`，然后 2D boolean mask 跨 blocks 提取。
- **`kv_indptr` 全 tensor 计算**：`cumsum` 替代 Python 逐步 append。

### 3.4 性能对比

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 循环迭代次数 | H_kv × num_active = 16 | H_kv = 8 |
| torch C++ 调用/chunk | 320-640 | **32** |
| `nonzero()` / `tolist()` 调用 | 16/chunk | **0** |
| Python while 循环 | 16/chunk | **0** |

---

## 4. 数据流总览

```
select_blocks (CPU):
  cos_matmul → top-p selection → selected_mask [H, G_k]
                                        │
                    ┌───────────────────────────────────┐
                    │  view() + copy_()  [O(1)]         │
                    │  → TensorSelection.mask            │
                    │    [H_kv, max_blocks, subs/block]  │
                    └───────────────────────────────────┘
                                        │
compute_chunked_prefill (CPU→GPU):      │
  for each pipeline chunk:              ▼
    ┌─────────────────────────────────────────────────┐
    │  repeat_interleave → token_mask                  │
    │  sum + cumsum → kv_indptr                        │
    │  for h in H_kv:                                  │
    │    fancy_index[bid_list, h] → k/v_blocks         │
    │    k/v_blocks[head_mask] → k/v_staging           │
    └─────────────────────────────────────────────────┘
                        │
                   H2D DMA transfer
                        │
                   GPU Jagged Kernel
```

---

## 5. 验证结果

测试环境：RTX 3090, Llama-3.1-8B-Instruct, 32k context, COMPASS policy.

| 测试 | 准确率 | Prefill 时间 |
|------|--------|-------------|
| 正确性 (top_p=1.0, λ=1e-10) | 100% | 11.2s |
| 性能 (top_p=0.9, λ=0.0001) | 100% | 10.7s |

nsys profile 文件：`results/nsys/COMPASS_offload_32k_blk4096_20260321_102711.nsys-rep`

---

## 6. 涉及文件

| 文件 | 变更说明 |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | 新增 `TensorSelection` dataclass |
| `nanovllm/kvcache/sparse/compass.py` | 向量化 `compass_build_entries` 和 `compass_v2_regroup` |
| `nanovllm/kvcache/offload_engine.py` | 重写 `gather_packed_from_mask()` |
