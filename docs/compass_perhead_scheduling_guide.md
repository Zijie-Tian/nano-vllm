# COMPASS Per-Head KV Cache Scheduling

## 1. Overview

COMPASS (Compacted Sub-block Sparse Attention) 的核心创新是 **per-head KV cache 调度**：不同 KV head 独立选择、独立传输、独立计算注意力，而非传统方式中"一个 head 命中就加载所有 head"。

### 1.1 动机

在 GQA (Grouped-Query Attention) 模型中（如 GLM-4-9B 的 4 KV heads + 32 Q heads），不同 KV head 的注意力分布差异显著。传统 union-all 方式取所有 head 选中 sub-block 的并集，导致大量不必要的 IO：

```
传统 Union-All:
  H0 选中: {sub-block 2, 5, 7}
  H1 选中: {sub-block 1, 3}
  H2 选中: {sub-block 5, 8}
  H3 选中: {sub-block 3, 6}
  → Union = {1, 2, 3, 5, 6, 7, 8} → 7 blocks × 4 heads = 28 head-blocks 传输

Per-Head:
  H0: 3 head-blocks, H1: 2, H2: 2, H3: 2 = 9 head-blocks 传输
  → IO 节省: 1 - 9/28 = 67.9%
```

### 1.2 实测 IO 节省

| top_p | Per-Head IO (sub-blocks) | Union-All IO (head-blocks) | IO 节省 | 准确率 |
|-------|--------------------------|----------------------------|---------|--------|
| 0.5   | 445                      | 576                        | 22.7%   | 100%   |
| 0.3   | 268                      | 392                        | 31.6%   | 100%   |

> [!TIP]
> top_p 越低（稀疏度越高），per-head 调度节省的 IO 比例越大，因为不同 head 的选择差异更明显。

---

## 2. 架构设计

### 2.1 数据流总览

```
┌─────────────────────────────────────────────────────────────┐
│                    select_blocks (CPU)                       │
│                                                             │
│  Q_pooled[G_q, H_kv, D] × K_packed[G_k, D/4]              │
│       ↓ T-MAC qGEMM (per KV head)                          │
│  scores[H_kv, G_q, G_k]                                    │
│       ↓ softmax + top-p                                     │
│  selected_mask[H_kv, G_k]  (bool)                          │
│       ↓ group by (head, block_id)                           │
│  PerHeadSubBlockSelection                                   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              compute_chunked_prefill (GPU)                   │
│                                                             │
│  for each KV head h:                                        │
│    for each batch of sub-blocks:                            │
│      ① CPU gather → staging[:, h, :]                       │
│      ② H2D: staging → GPU slot                             │
│      ③ GPU: BLASST(q_h, k_h, v_h) → merge within head     │
│    → head_o[1, q_len, gqa_ratio, D]                        │
│  cat across heads → historical_o[1, q_len, num_heads, D]   │
│  merge with current chunk → final output                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据结构

#### `PerHeadSubBlockSelection`

```python
@dataclass
class PerHeadSubBlockSelection:
    per_head_entries: List     # [H][(cpu_block_id, [sub_block_indices])]
    sub_block_size: int = 128  # tokens per sub-block
    num_kv_heads: int = 1

    # Derived properties:
    per_head_num_subblocks: List  # [H] int counts
    per_head_tokens: List         # [H] token counts (= num_subblocks × sub_block_size)
    total_subblocks: int          # sum across all heads
    total_tokens: int             # total_subblocks × sub_block_size
```

**示例**（GLM-4-9B, 4 KV heads, top_p=0.3, chunk 7）:
```
per_head_entries[0] = [(blk_3, [2, 5, 7]), (blk_5, [1, 3]), ...]  # H0: 67 sub-blocks
per_head_entries[1] = [(blk_2, [0, 4]), (blk_5, [6]), ...]         # H1: 67 sub-blocks
per_head_entries[2] = [(blk_1, [3]), (blk_3, [5, 8]), ...]         # H2: 67 sub-blocks
per_head_entries[3] = [(blk_4, [2, 3, 6]), ...]                    # H3: 67 sub-blocks
```

---

## 3. 内存布局

### 3.1 CPU KV Cache

```
k_cache_cpu[num_layers, num_blocks, block_size=4096, kv_heads=4, head_dim=128]
```

每个 block 包含 4096 tokens × 4 heads，每 128 tokens 为一个 sub-block（共 32 sub-blocks/block）。

### 3.2 Staging Buffer (Pinned CPU)

```
staging_k_cpu[block_size=4096, kv_heads=4, head_dim=128]
```

Per-head gather 将每个 head 的选中 sub-blocks **沿 token 轴紧密打包**到对应 head 列：

```
                   H0         H1         H2         H3
Token 0-127    │ H0-sub0  │ H1-sub0  │ H2-sub0  │ H3-sub0  │
Token 128-255  │ H0-sub1  │ H1-sub1  │ H2-sub1  │ (unused) │
Token 256-383  │ H0-sub2  │ (unused) │ (unused) │ (unused) │
               │  ...     │          │          │          │
```

**关键特性**：
- 不同 head 的有效 token 数不同（`per_head_tokens[h]`）
- H2D 传输 `max(per_head_tokens)` 行（包含所有 head 的数据）
- GPU 计算时按 `per_head_tokens[h]` 截取特定 head

### 3.3 GPU Ring Buffer Slot

```
k_cache_gpu[slot, block_size=4096, kv_heads=4, head_dim=128]
```

GPU 端从 slot 中读取单个 head 的数据：
```python
k_h = k_cache_gpu[slot, :per_head_tokens[h], h, :]  # [n_h, head_dim]
```

---

## 4. 关键算法

### 4.1 select_blocks: 两级稀疏选择

**Level 1 (CPU)**：T-MAC qGEMM 估计 + top-p 选择

```python
# 1. GPU 端 Q 池化 + GQA fold
q_pooled = pool_and_normalize(q, stride=128)  # [G_q, H_kv, D]

# 2. Async 传输到 CPU (~128KB)
q_cpu = async_copy_to_cpu(q_pooled)

# 3. CPU 端 T-MAC 估计 (per KV head)
for h in range(H_kv):
    for gk in range(G_k):  # 遍历所有 K sub-blocks
        score[h, gk] = tmac_qgemm(q_cpu[:, h, :], k_packed[gk])

# 4. Per-head softmax + top-p 选择
for h in range(H_kv):
    probs = softmax(score[h])         # [G_k]
    cumsum = cumulative_sum(sorted(probs, desc=True))
    selected_mask[h] = probs ∈ top-p   # 独立选择
```

### 4.2 compute_chunked_prefill: Per-Head 注意力计算

```python
per_head_mgin = [None] * kv_heads   # m_global 每个 chunk 重新开始！

for h in range(kv_heads):
    flat_subs = flatten(selection.per_head_entries[h])
    head_o, head_lse = None, None
    mgin_h = None                    # ← 不跨 chunk 持久化

    for batch in split(flat_subs, max_subs_per_batch):
        # ① CPU Gather
        gather_subblocks_per_head(layer_id, [batch], head_offset=h)

        # ② H2D Transfer
        load_staging_to_slot(slot=0, max_tokens)
        wait_slot_layer(slot=0)

        # ③ GPU Compute (single KV head)
        k_h = gpu_slot[:batch_tokens, h, :]    # [n, D]
        q_h = q[:, h*gqa : (h+1)*gqa, :, :]   # [1, gqa, q_len, D]
        out, lse, mg = BLASST(q_h, k_h.unsqueeze, ...)
        mgin_h = mg   # running max within this chunk

        # ④ Merge within head (across batches)
        head_o, head_lse = merge(head_o, head_lse, out, lse)

    per_head_o_list.append(head_o)   # [1, q_len, gqa_ratio, D]
    per_head_lse_list.append(head_lse)  # [1, gqa_ratio, q_len]

# ⑤ Concatenate across heads
historical_o = cat(per_head_o_list, dim=2)    # [1, q_len, num_heads, D]
historical_lse = cat(per_head_lse_list, dim=1) # [1, num_heads, q_len]

# ⑥ 重组 m_global 传递给 current chunk
mg_parts = [per_head_mgin[h] for h in range(kv_heads)]
historical_m_global = cat(mg_parts, dim=1)  # [1, num_heads, q_len]

# ⑦ Current chunk BLASST (causal, with historical m_global)
out_curr, lse_curr, _ = BLASST(q, k_curr, v_curr,
                                m_global_in=historical_m_global,
                                is_causal=True, kv_offset=...)

# ⑧ Final merge
output = merge(historical_o, historical_lse, out_curr, lse_curr)
```

---

## 5. 同步模型

### 5.1 事件链路

```
gather (CPU sync) → load_staging_to_slot (transfer_stream async)
                         │
                         ├─ wait_event(compute_done[slot])
                         ├─ wait_event(offload_done[slot])
                         ├─ .copy_(non_blocking=True)
                         └─ ring_slot_ready[slot].record()
                                     │
                    wait_slot_layer ──┘
                         │
                         └─ compute_stream.wait_event(ready[slot])
                                     │
                    BLASST compute ───┘   (on compute_stream)
                         │
                    compute_stream.synchronize()
                         │
                    record_slot_compute_done(slot)  (on default stream)
```

### 5.2 Staging Buffer 安全性

Per-head 路径使用 `slot=0` 做所有 H2D 传输。在每次循环中：

1. **compute_stream.synchronize()** 确保 GPU 端已完成对 slot 数据的读取
2. `record_slot_compute_done(0)` 在 default stream 上记录完成事件
3. 下一次 `load_staging_to_slot` 的 `wait_event(compute_done[0])` 确保不会在计算完成前覆盖 slot

**Staging buffer vs CPU cache**：`load_to_slot_layer`（BLASST 原始路径）直接从不可变的 `k_cache_cpu` 复制到 GPU，不存在 staging buffer 竞争。Per-head 路径经过 staging buffer，但因为使用单 slot 串行执行，每次 gather 前上一次的 H2D 已完成。

---

## 6. 已解决的 Bug

### 6.1 m_global 跨 Chunk 持久化 (Critical)

**症状**：per-head 路径输出全零，准确率 0%。

**根因**：`_per_head_m_global[layer_id][h]` 将 chunk N 的 m_global 持久化到 chunk N+1。BLASST kernel 使用 m_global 做动态剪枝：

```python
if m_local - m_global < ln(λ):
    skip this sub-block  # 不做 Softmax + PV
```

chunk N 产生的高 m_global 带入 chunk N+1 后，新 Q tokens 的注意力分数 `m_local` 远低于残留阈值，导致所有历史 sub-block 被跳过。

**修复**：每次 `compute_chunked_prefill` 调用时用局部变量 `per_head_mgin = [None] * kv_heads` 替代持久化 dict。与 BLASST 参考实现一致，m_global 在每个 chunk 起始为 None (= -inf)。

### 6.2 零 Head Fallback Shape 不匹配

**症状**：当某个 head 选中 0 个 sub-block 时，fallback tensor 的 shape 与正常计算结果不一致。

**根因**：fallback 创建 `[1, gqa_ratio, q_len, D]`，但正常路径经过 `transpose(1,2)` 产生 `[1, q_len, gqa_ratio, D]`。

**修复**：fallback shape 改为 `[1, q_len, gqa_ratio, D]`。

---

## 7. 性能特征

### 7.1 IO 行为（32K context, GLM-4-9B, chunk 7）

| top_p | Per-Head sub-blocks | Union sub-blocks | Union×H (旧 IO) | IO 比率 |
|-------|---------------------|------------------|-----------------|---------|
| 0.99  | 896 (100%)          | 224 (100%)       | 896             | 100%    |
| 0.5   | 445 (49.7%)         | 144 (64.3%)      | 576             | 77.3%   |
| 0.3   | 268 (29.9%)         | 98 (43.8%)       | 392             | 68.4%   |

> [!NOTE]
> Per-Head IO 比率 = per_head_total / (union_count × H)。该比率随 top_p 降低而降低，因为 head 间选择差异增大。

### 7.2 计算特征

- **单 head BLASST**：每次 BLASST 调用处理 `gqa_ratio` 个 Q heads × 1 KV head
- **Batch 大小**：受 staging buffer 容量 (block_size=4096 tokens) 限制
- **串行执行**：当前使用 slot=0 串行处理，无 pipeline overlap

### 7.3 已知的性能瓶颈

1. **无 ring buffer pipeline**：所有 head/batch 串行使用 slot=0，每次等待 H2D 完成后才计算
2. **单 head BLASST 调用开销**：`4 heads × ceil(n_h / block_size) batches` 次 BLASST kernel launch
3. **多次 gather**：每个 head 的每个 batch 都执行一次 CPU gather

**优化方向**：
- 使用多个 ring buffer slot 实现 head 间 pipeline
- 合并多个 head 的 gather 到一次操作中
- 探索 per-head 数据的 batch BLASST kernel

---

## 8. API 参考

### 8.1 `OffloadEngine.gather_subblocks_per_head`

```python
def gather_subblocks_per_head(
    self,
    layer_id: int,
    per_head_selections: List[List],  # [H][(bid, [si...])]
    sub_block_size: int = 128,
    head_offset: int = 0,
) -> Tuple[int, List[int]]:
```

**语义**：将每个 head 选中的 sub-blocks 拷贝到 staging buffer 的对应 head 列。

**参数**：
- `per_head_selections[h_local]`：head `h_local + head_offset` 的选中 sub-blocks
- `head_offset`：当单 head 调用时，传入实际 KV head 索引

**返回**：
- `max_tokens`：所有 head 中最大的 token 数（用于 H2D 传输量控制）
- `per_head_tokens`：每个 head 的 token 数

### 8.2 `OffloadEngine.load_staging_to_slot`

```python
def load_staging_to_slot(
    self,
    slot_idx: int,
    num_tokens: int,
    layer_id: int = -1,
    is_prefill: bool = True,
) -> None:
```

**语义**：Async H2D 传输 `staging[:num_tokens]` → `gpu_slot[:num_tokens]`（保持 `[tokens, heads, D]` 布局）。

### 8.3 `COMPASSPolicy.compute_chunked_prefill`

详见 [compass.py](../nanovllm/kvcache/sparse/compass.py) 中的实现。关键签名：

```python
def compute_chunked_prefill(
    self,
    q: torch.Tensor,           # [q_len, num_heads, head_dim]
    offload_engine,
    kvcache_manager,
    layer_id: int,
    current_chunk_idx: int,
    num_tokens: int,
    selected_blocks: List[int],
) -> torch.Tensor:             # [q_len, num_heads, head_dim]
```

---

## 9. 相关文档

- [COMPASS 分析](compass_analysis.md)：GPU Q pooling、CPU profiling、top_p sweep
- [COMPASS sub-block compacted transfer](compass_subblock_compacted_transfer.md)：gather + bulk H2D pipeline
- [BLASST 性能分析](blasst_performance_analysis.md)：λ vs density、精度稳定性
- [BLASST per-head density analysis](blasst_perhead_density_analysis.md)：per-head KV density 分布分析
- [Sparse policy architecture](sparse_policy_architecture.md)：SparsePolicy 抽象层设计

---

**Author**: Zijie Tian / Antigravity  
**Created**: 2026-03-16  
**Status**: Verified (100% accuracy at top_p=0.5 and top_p=0.3, 32K context, GLM-4-9B)
