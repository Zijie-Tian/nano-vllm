# XAttention KV Chunking Kernels

## 概述

本文档描述了支持 KV 维度分 chunk 的 softmax kernels 实现。这些 kernels 允许在 CPU offload 场景下，沿 KV 维度分块计算 sparse attention estimation，而不需要在 GPU 上保存完整的 raw attention scores。

## 背景

原始的 `softmax_fuse_block_sum` kernel 需要完整的 K 序列来计算正确的 softmax 归一化分母：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

如果只有部分 K (KV chunk)，分母 `Σ_j exp(x_j)` 不完整，导致归一化错误。

## 解决方案：三阶段计算

通过将 softmax 计算拆分为三个阶段，实现正确的 KV chunking：

### 阶段 1: `softmax_compute_partial_stats`

计算每个 KV chunk 的 partial statistics：
- `m_partial`: 该 chunk 内的最大值 (per query row)
- `l_partial`: 该 chunk 内的 partial sum = Σ exp(x - m_partial)

```python
m_partial, l_partial = softmax_compute_partial_stats(
    attn_weights_kv,      # [batch, heads, q_len, k_chunk_len]
    reshaped_block_size,
    segment_size,
    scale,
    chunk_start=chunk_start,
    kv_offset=kv_offset,  # KV chunk 在完整序列中的偏移
    is_causal=True,
)
# 输出: m_partial, l_partial 形状为 [batch, heads, q_len]
```

### 阶段 2: `merge_softmax_stats`

Host 端合并所有 KV chunks 的 statistics：

```python
m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)
```

合并公式 (Online Softmax):
```
m_new = max(m_global, m_chunk)
l_new = l_global * exp(m_global - m_new) + l_chunk * exp(m_chunk - m_new)
```

### 阶段 3: `softmax_normalize_and_block_sum`

使用全局 statistics 归一化并计算 block sums：

```python
attn_sum_kv = softmax_normalize_and_block_sum(
    attn_weights_kv,      # [batch, heads, q_len, k_chunk_len]
    m_global,             # [batch, heads, q_len]
    l_global,             # [batch, heads, q_len]
    reshaped_block_size,
    segment_size,
    chunk_start=chunk_start,
    real_q_len=real_q_len,
    scale=scale,
    kv_offset=kv_offset,
    is_causal=True,
)
# 输出: [batch, heads, q_blocks, k_chunk_blocks]
```

## 数学等价性证明

原始 softmax 计算 (完整 K):
```
softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)
```

分 KV chunk 计算:
```
Chunk 0: m_0 = max(x[0:N/2]), l_0 = Σ exp(x[0:N/2] - m_0)
Chunk 1: m_1 = max(x[N/2:N]), l_1 = Σ exp(x[N/2:N] - m_1)

合并:
m_global = max(m_0, m_1)
l_global = l_0 * exp(m_0 - m_global) + l_1 * exp(m_1 - m_global)
         = Σ exp(x[0:N] - m_global)  # 等于全局 sum

归一化:
softmax(x_i) = exp(x_i - m_global) / l_global  # 正确!
```

## Causal Mask 处理

两个 kernel 都正确处理了 causal attention：

1. **`softmax_partial_stats_kernel`**: 通过 `kv_offset` 参数确定当前 KV chunk 在完整序列中的位置，正确计算 causal boundary

2. **`softmax_normalize_block_sum_kernel`**: 同样使用 `kv_offset`，对 causal boundary 之后的位置输出 0

## API 参考

### `softmax_compute_partial_stats`

```python
def softmax_compute_partial_stats(
    attn_weights_slice: torch.Tensor,  # [batch, heads, q_len, k_chunk_len]
    reshaped_block_size: int,
    segment_size: int,
    scale: float,
    chunk_start: int = 0,              # Q chunk 起始位置 (reshaped space)
    kv_offset: int = 0,                # KV chunk 偏移 (reshaped space)
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (m, l) partial stats"""
```

### `merge_softmax_stats`

```python
def merge_softmax_stats(
    m_chunks: list,  # List of [batch, heads, q_len] tensors
    l_chunks: list,  # List of [batch, heads, q_len] tensors
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (m_global, l_global)"""
```

### `softmax_normalize_and_block_sum`

```python
def softmax_normalize_and_block_sum(
    attn_weights_slice: torch.Tensor,  # [batch, heads, q_len, k_chunk_len]
    m_global: torch.Tensor,            # [batch, heads, q_len]
    l_global: torch.Tensor,            # [batch, heads, q_len]
    reshaped_block_size: int,
    segment_size: int,
    chunk_start: int,
    real_q_len: int,
    scale: float,
    kv_offset: int = 0,
    is_causal: bool = False,
) -> torch.Tensor:
    """返回 block sums [batch, heads, q_blocks, k_chunk_blocks]"""
```

## 使用示例

```python
from nanovllm.ops.xattn import (
    flat_group_gemm_fuse_reshape,
    softmax_compute_partial_stats,
    softmax_normalize_and_block_sum,
    merge_softmax_stats,
    find_blocks_chunked,
)

# 对每个 Q chunk
for q_chunk_idx in range(q_chunk_num):
    Q_chunk = Q_padded[:, :, q_start:q_end, :]

    # 阶段 1: 每个 KV chunk 计算 partial stats
    m_chunks, l_chunks = [], []
    attn_weights_chunks = []

    for kv_chunk_idx in range(kv_chunk_num):
        K_chunk = K_padded[:, :, kv_start:kv_end, :]
        kv_offset = kv_chunk_idx * kv_chunk_size // STRIDE

        # 计算 raw scores
        attn_weights = flat_group_gemm_fuse_reshape(
            Q_chunk, K_chunk, STRIDE,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_causal=False,  # K 不完整
        )
        attn_weights_chunks.append(attn_weights)

        # 计算 partial stats
        m, l = softmax_compute_partial_stats(
            attn_weights, block_size, segment_size, scale,
            chunk_start=chunk_start,
            kv_offset=kv_offset,
            is_causal=True,
        )
        m_chunks.append(m)
        l_chunks.append(l)

    # 阶段 2: 合并 stats
    m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)

    # 阶段 3: 归一化并计算 block sums
    block_sums_list = []
    for kv_chunk_idx, attn_weights in enumerate(attn_weights_chunks):
        kv_offset = kv_chunk_idx * kv_chunk_size // STRIDE
        block_sums = softmax_normalize_and_block_sum(
            attn_weights, m_global, l_global,
            block_size, segment_size, chunk_start, real_q_len, scale,
            kv_offset=kv_offset,
            is_causal=True,
        )
        block_sums_list.append(block_sums)

    # 拼接并选择 blocks
    attn_sum = torch.cat(block_sums_list, dim=-1)
    mask = find_blocks_chunked(attn_sum, ...)
```

## 性能对比

| 方面 | 原始实现 | KV Chunking 实现 |
|------|---------|-----------------|
| Kernel 数量 | 1 | 2 (stats + normalize) |
| Raw scores 读取次数 | 2 | 2 |
| 额外内存 | 0 | O(batch × heads × q_len × 2) for (m, l) |
| Host 计算 | 无 | merge stats (轻量) |
| **峰值显存** | O(q_len × k_full_len) | **O(q_len × k_chunk_len)** |

## 验证

测试脚本 `tests/test_xattn_estimate_alignment.py` 验证了 KV chunking 实现与原始 `xattn_estimate` API 的一致性：

```
| 方法 | density | 与 API 差异 | Mask 差异 |
|------|---------|-------------|-----------|
| xattn_estimate API | 0.159023 | - | - |
| KV chunking | 0.159023 | 0.000000 | 0.0044% |
```

## 相关文件

- `nanovllm/ops/xattn.py`: Kernel 实现
- `tests/test_xattn_estimate_alignment.py`: 验证测试
- `docs/xattn_kernels_guide.md`: 原始 kernel 文档
