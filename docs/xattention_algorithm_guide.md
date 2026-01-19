# XAttention 算法实现指南

本文档详细描述 COMPASS 项目中 XAttention 的算法原理和实现细节。

## 概述

XAttention 是一种基于 **stride reshape** 的块稀疏注意力方法，通过低成本估计识别重要块，然后使用 **BSA (Block Sparse Attention)** 库执行稀疏计算。

### 核心依赖

| 组件 | 来源 | 作用 |
|------|------|------|
| Triton Kernels | COMPASS 自研 | Q/K reshape + 块级估计 |
| BSA | MIT-HAN-LAB `block_sparse_attn` | 稀疏注意力计算 |

---

## 算法流程

```
输入: Q [batch, heads, q_len, head_dim]
      K [batch, heads, k_len, head_dim]
      V [batch, heads, k_len, head_dim]

┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Stride Reshape (inverse 模式)                       │
│                                                              │
│ K_reshaped = concat([K[:,:,k::stride,:] for k in stride])   │
│ Q_reshaped = concat([Q[:,:,(stride-1-q)::stride,:] for q])  │
│                                                              │
│ 效果: 序列长度从 seq_len 缩短到 seq_len/stride               │
│       head_dim 扩展到 head_dim * stride                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: 块级注意力估计 (Triton 加速)                         │
│                                                              │
│ 2a. flat_group_gemm_fuse_reshape:                           │
│     计算 Q_reshaped @ K_reshaped^T                          │
│     输出: attn_weights [batch, heads, q_len/stride, k_len/stride] │
│                                                              │
│ 2b. softmax_fuse_block_sum:                                 │
│     - 在线 softmax (数值稳定)                                │
│     - 按 block_size/stride 分组求和                          │
│     输出: attn_sum [batch, heads, q_blocks, k_blocks]        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 块选择 (find_blocks_chunked)                        │
│                                                              │
│ 对每个 Q block:                                              │
│   1. 按 attn_sum 降序排序 K blocks                           │
│   2. 累积求和直到 >= threshold * total_sum                   │
│   3. 累积到的 blocks 标记为 True                             │
│                                                              │
│ 特殊处理:                                                    │
│   - 对角块 (causal) 始终保留                                 │
│   - Sink 块 (block 0) 可选保留                               │
│                                                              │
│ 输出: simple_mask [batch, heads, q_blocks, k_blocks] (bool)  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: 稀疏注意力计算 (BSA)                                 │
│                                                              │
│ attn_output = block_sparse_attn_func(                       │
│     Q, K, V,                                                 │
│     q_cu_seq_lens,      # [0, q_len]                        │
│     k_cu_seq_lens,      # [0, k_len]                        │
│     head_mask_type,     # [num_heads] 全 1                   │
│     None,               # left_mask                          │
│     simple_mask,        # 块稀疏 mask                        │
│     q_len, k_len,                                            │
│     is_causal=True,                                          │
│ )                                                            │
│                                                              │
│ 输出: attn_output [batch, heads, q_len, head_dim]            │
└─────────────────────────────────────────────────────────────┘
```

---

## Stride Reshape 详解

### Inverse 模式

XAttention 默认使用 `select_mode="inverse"`，这是一种交错采样策略：

```python
# 原始: Q/K shape = [batch, heads, seq_len, head_dim]
# stride = 8

# K reshape: 正向交错
K_reshaped = concat([K[:, :, 0::8, :],   # 位置 0, 8, 16, ...
                     K[:, :, 1::8, :],   # 位置 1, 9, 17, ...
                     K[:, :, 2::8, :],   # 位置 2, 10, 18, ...
                     ...
                     K[:, :, 7::8, :]])  # 位置 7, 15, 23, ...
# 结果: [batch, heads, seq_len/8, head_dim * 8]

# Q reshape: 反向交错 (inverse)
Q_reshaped = concat([Q[:, :, 7::8, :],   # 位置 7, 15, 23, ...
                     Q[:, :, 6::8, :],   # 位置 6, 14, 22, ...
                     Q[:, :, 5::8, :],   # 位置 5, 13, 21, ...
                     ...
                     Q[:, :, 0::8, :]])  # 位置 0, 8, 16, ...
# 结果: [batch, heads, seq_len/8, head_dim * 8]
```

### 为什么用 Inverse 模式？

当计算 `Q_reshaped @ K_reshaped^T` 时，inverse 模式使得：
- Q 的后半部分与 K 的前半部分对齐
- 这样可以近似捕获 **causal attention 的对角模式**

---

## Triton Kernels 详解

### 1. flat_group_gemm_fuse_reshape

**文件**: `compass/src/kernels.py:198-235`

**功能**: 融合 stride reshape 和 GEMM，避免显式创建 reshape 后的大张量

```python
@triton.jit
def flat_group_gemm_fuse_reshape_kernel(Q, K, Out, ...):
    # 关键: 不实际 reshape，而是通过指针算术模拟
    Q_ptrs = Q + block_m * BLOCK_M * STRIDE * stride_qn
    K_ptrs = K + block_n * BLOCK_N * STRIDE * stride_kn

    # 对 stride 个位置累加
    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)  # Q inverse 采样
        k = tl.load(K_ptrs + iter * stride_kn)  # K 正向采样
        o += tl.dot(q, k)
```

**优势**:
- 内存节省: 不需要创建 `[batch, heads, seq_len/stride, head_dim*stride]` 的中间张量
- 计算融合: reshape + GEMM 一次完成

### 2. softmax_fuse_block_sum

**文件**: `compass/src/kernels.py:6-95`

**功能**: 在线 softmax + 块内求和

```python
@triton.jit
def softmax_fuse_block_sum_kernel_causal(In, Out, ...):
    # Pass 1: 计算全局 max 和 sum (在线算法)
    for iter in range(num_iters):
        X = tl.load(input_ptr + iter * segment_size) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    # Pass 2: 归一化并按块求和
    for iter in range(num_iters):
        X = tl.load(input_ptr + iter * segment_size) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]  # softmax
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2).sum(0)  # 块内求和
        tl.store(output_ptr + iter * segment_size // block_size, X)
```

**输出含义**: `attn_sum[b, h, qi, ki]` = Q block qi 对 K block ki 的**归一化注意力权重之和**

---

## 块选择算法 (find_blocks_chunked)

**文件**: `compass/src/utils.py:44-191`

### 算法步骤

```python
def find_blocks_chunked(input_tensor, current_index, threshold, ...):
    """
    input_tensor: [batch, heads, q_blocks, k_blocks] - 块级注意力权重和
    threshold: 0.9 - 累积阈值
    """
    # 1. 计算每行总和
    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    required_sum = total_sum * threshold  # 需要达到的累积和

    # 2. 特殊块始终保留
    mask = zeros_like(input_tensor, dtype=bool)
    mask[:, :, :, 0] = True              # sink 块
    mask[:, :, :, diagonal] = True       # 对角块 (causal)

    # 3. 对剩余块按权重排序
    other_values = input_tensor.masked_fill(mask, 0)
    sorted_values, index = sort(other_values, descending=True)

    # 4. 累积求和直到达到阈值
    cumsum = sorted_values.cumsum(dim=-1)
    index_mask = cumsum < required_sum

    # 5. 标记选中的块
    mask[..., index[index_mask]] = True

    return mask
```

### 示例

```
threshold = 0.9
attn_sum 某一行 = [0.05, 0.30, 0.40, 0.15, 0.10]  (已 softmax, 和为 1.0)
required_sum = 0.9

排序后: [0.40, 0.30, 0.15, 0.10, 0.05]
累积和: [0.40, 0.70, 0.85, 0.95, 1.00]
                            ↑ 达到 0.9

选中: 前 4 个块 (indices: 2, 1, 3, 4)
```

---

## BSA (Block Sparse Attention)

### 库来源

```python
from block_sparse_attn import block_sparse_attn_func
```

来自 MIT-HAN-LAB，提供基于块 mask 的高效稀疏 FlashAttention 实现。

### 接口

```python
attn_output = block_sparse_attn_func(
    query_states,         # [total_q, num_heads, head_dim]
    key_states,           # [total_k, num_heads, head_dim]
    value_states,         # [total_k, num_heads, head_dim]
    q_cu_seq_lens,        # [batch+1] cumulative sequence lengths
    k_cu_seq_lens,        # [batch+1]
    head_mask_type,       # [num_heads] int32, 1=causal, 0=full
    left_mask,            # Optional left padding mask
    block_mask,           # [batch, heads, q_blocks, k_blocks] bool
    max_seqlen_q,         # int
    max_seqlen_k,         # int
    p_dropout=0.0,
    deterministic=True,
    is_causal=True,       # 全局 causal flag
)
```

### 块大小要求

BSA 要求 **block_size = 128**（硬编码）：
```python
assert block_size == 128  # Xattention.py:358
```

---

## 关键参数

| 参数 | 默认值 | 范围 | 作用 |
|------|--------|------|------|
| `stride` | 8 | 4-16 | Q/K 交错采样步长，越大估计越快但越粗糙 |
| `threshold` | 0.9 | 0.7-0.99 | 累积注意力阈值，越高保留块越多 |
| `block_size` | 128 | 128 (固定) | BSA 块大小，不可调 |
| `chunk_size` | 16384 | 2048-131072 | 估计时的分块大小，影响内存使用 |
| `norm` | 1.0 | 0.5-2.0 | 注意力分数归一化系数 |
| `keep_sink` | False | bool | 是否始终保留第一个块 |
| `keep_recent` | False | bool | 是否始终保留对角块 |

---

## 计算复杂度

### 估计阶段

| 操作 | 复杂度 |
|------|--------|
| Stride reshape GEMM | O(seq_len/stride × seq_len/stride × head_dim × stride) = O(seq_len² × head_dim / stride) |
| Softmax + block sum | O(seq_len² / stride²) |
| Block selection | O(num_blocks² × log(num_blocks)) |

**估计阶段总复杂度**: O(seq_len² × head_dim / stride)

### 计算阶段 (BSA)

设选中块比例为 ρ (通常 0.3-0.5):

| 操作 | 复杂度 |
|------|--------|
| Block sparse attention | O(ρ × num_blocks² × block_size² × head_dim) = O(ρ × seq_len² × head_dim) |

**总复杂度**: O(seq_len² × head_dim × (1/stride + ρ))

当 stride=8, ρ=0.4 时，相比 full attention 节省约 **50%** 计算量。

---

## 与 nano-vllm 集成注意事项

### 依赖要求

```
block_sparse_attn  # pip install block-sparse-attn
triton >= 2.0      # Triton kernels
```

### CPU Offload 场景适配

XAttention 原始实现假设所有 KV 在 GPU 上。对于 CPU offload 场景，需要：

1. **估计阶段**: 仍需加载所有历史 KV 到 GPU 进行估计
2. **计算阶段**: 只加载选中的块

这可能需要修改为两阶段 pipeline:
- 先用采样数据估计重要块
- 再只加载重要块进行计算

### block_size 对齐

nano-vllm 的 `kvcache_block_size` 需要与 BSA 的 128 对齐：
- 如果 `kvcache_block_size = 1024`，则每个 kv block 包含 8 个 BSA blocks
- 块选择粒度需要相应调整

---

## 源文件索引

| 文件 | 位置 | 内容 |
|------|------|------|
| `Xattention.py` | `compass/src/Xattention.py` | 主入口: `xattn_estimate()`, `Xattention_prefill()` |
| `kernels.py` | `compass/src/kernels.py` | Triton 内核 |
| `utils.py` | `compass/src/utils.py` | `find_blocks_chunked()`, `create_causal_mask()` |

---

## 参考

- COMPASS 项目: `/home/zijie/Code/COMPASS/`
- BSA 库: MIT-HAN-LAB block_sparse_attn
- 测试报告: `docs/xattention_bsa_test_report.md`
