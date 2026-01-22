# XAttention Kernels Guide

本文档详细说明 XAttention 的两个核心 Triton kernel 的工作原理。

## 概述

XAttention 使用 stride 采样来快速估计 attention 分布，用于稀疏 attention 的 block 选择。

**数据流**：
```
Q [batch, heads, q_len, head_dim]
K [batch, heads, kv_len, head_dim]
  ↓ flat_group_gemm_fuse_reshape (stride 采样 + GEMM)
attn_scores [batch, heads, q_len/stride, kv_len/stride]
  ↓ softmax_fuse_block_sum (softmax + block 求和)
block_sums [batch, heads, q_blocks, k_blocks]
  ↓ threshold 选择
sparse_mask [batch, heads, q_blocks, k_blocks]
```

**注意**：Q 和 K 可以有不同的长度（q_len ≠ kv_len），这在 chunked prefill 场景中很常见。

## Kernel 1: flat_group_gemm_fuse_reshape

### 功能

计算 stride reshape 后的 attention scores，本质是计算原始 attention 矩阵中每个 stride×stride 块的**反对角线求和**。

### 函数签名

```python
def flat_group_gemm_fuse_reshape(
    query_states: torch.Tensor,  # [batch, heads, q_len, head_dim]
    key_states: torch.Tensor,    # [batch, heads, kv_len, head_dim]
    stride: int,
    chunk_start: int,
    chunk_end: int,
    is_causal: bool = True,
) -> torch.Tensor:  # [batch, heads, q_len/stride, kv_len/stride]
```

### 采样方式

```
Q 采样: (stride-1-s)::stride  (逆向)
K 采样: s::stride             (正向)

例如 stride=4:
  Q 采样位置: 3, 7, 11, 15, ...  (从位置 3 开始，每隔 4)
  K 采样位置: 0, 4, 8, 12, ...   (从位置 0 开始，每隔 4)
```

### 反对角线原理

对于原始 attention 矩阵的每个 stride×stride 块：

```
stride=4 的块:
     K[0]  K[1]  K[2]  K[3]
Q[0]  ·     ·     ·     X    ← 反对角线
Q[1]  ·     ·     X     ·
Q[2]  ·     X     ·     ·
Q[3]  X     ·     ·     ·
```

**输出值 = 反对角线元素之和**

因为：
- `Q[i]` 采样自原始位置 `(stride-1-i)`
- `K[j]` 采样自原始位置 `j`
- 当 `i + j = stride - 1` 时，恰好在反对角线上

### Triton 约束

**GPU 相关的 BLOCK 大小**：

| GPU 类型 | 显存 | BLOCK_M/N | 最小 q_len/kv_len |
|----------|------|-----------|-------------------|
| RTX 3090 | 24GB | 64 | stride × 64 = 256 |
| A100/H100 | ≥40GB | 128 | stride × 128 = 512 |

```python
# 代码中的判断逻辑
if props.total_memory < 30 * 1024**3:  # < 30GB
    BLOCK_M = BLOCK_N = 64
else:
    BLOCK_M = BLOCK_N = 128

assert q_len % (stride * BLOCK_M) == 0
assert kv_len % (stride * BLOCK_N) == 0
```

### 验证示例

```python
# 输入: 偶数位置=1, 奇数位置=2
# q_len=512, kv_len=2048, stride=4, head_dim=128

# 反对角线元素 (stride=4):
#   Q[奇]*K[偶] + Q[偶]*K[奇] = 2*1 + 1*2 = 4 (每对)
#   stride=4 有 2 对
#   乘以 head_dim=128
# 预期值: 4 * 2 * 128 = 1024

# 输出 shape: [1, 1, 128, 512]  (512/4=128, 2048/4=512)
```

## Kernel 2: softmax_fuse_block_sum

### 功能

对 `flat_group_gemm_fuse_reshape` 的输出做 softmax，然后按 block 求和，得到每个 block 的 attention 权重总和。

### 参数说明

| 参数 | 含义 |
|------|------|
| `attn_weights_slice` | 输入 attention scores `[batch, heads, q_reshaped, k_reshaped]` |
| `reshaped_block_size` | Block 大小（在 reshaped 空间，= block_size / stride） |
| `segment_size` | 每次迭代处理的 K 维度大小（tiling） |
| `chunk_start` | Q 的起始位置（用于 causal mask） |
| `chunk_end` | Q 的结束位置 |
| `real_q_len` | 有效 Q 长度（用于 padding mask） |
| `scale` | 缩放因子（融合多个因素） |
| `is_causal` | 是否应用 causal mask |

### Scale 因子

```python
scale = log2(e) / sqrt(head_dim) / stride / norm
     = 1.4426950408889634 / sqrt(head_dim) / stride / norm
```

| 因子 | 值 | 作用 |
|------|-----|------|
| `log2(e)` | 1.4426950408889634 | Triton 用 `exp2` 而非 `exp`，需转换底数 |
| `1/sqrt(head_dim)` | 1/√128 | 标准 attention 缩放 |
| `1/stride` | 1/4 | stride 采样的归一化 |
| `1/norm` | 变化 | 额外归一化因子 |

**为什么用 exp2**：Triton 的 `exp2` 比 `exp` 更快（硬件原生支持），所以把 log₂(e) 融合到 scale 里。

### Segment Size 约束

```python
assert segment_size >= reshaped_block_size
```

原因：kernel 内部使用 `segment_size // block_size` 做 reshape：

```python
X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
```

如果 `segment_size < block_size`，则 `segment_size // block_size = 0`，导致无效维度。

### 验证示例

```python
# 输入: attn_scores [1, 1, 128, 512] (所有值相同)
# block_size=128

# softmax 后每行均匀分布 (所有值相同 → 均匀)
# 每行对一个 K block 的贡献 = block_size / kv_reshaped_len = 128/512 = 0.25
# 每个 Q block 有 block_size=128 行
# block_sum = 128 * 0.25 = 32

# 输出 shape: [1, 1, 1, 4]  (128/128=1, 512/128=4)
```

## 完整示例

```python
# 参数
q_len = 512       # Q 长度
kv_len = 2048     # K/V 长度 (可以不同于 q_len)
stride = 4
block_size = 128

# Step 1: flat_group_gemm_fuse_reshape
# 输入: Q [1,1,512,128], K [1,1,2048,128]
# 输出: attn_scores [1,1,128,512]

# Step 2: softmax_fuse_block_sum
# 输入: attn_scores [1,1,128,512]
# 输出: block_sums [1,1,1,4]
#       q_blocks = 128/128 = 1
#       k_blocks = 512/128 = 4
```

## 测试代码

参考 `tests/test_xattn_kernels.py`，使用结构化数据验证两个 kernel 的正确性。

## 相关文档

- [`docs/xattention_algorithm_guide.md`](xattention_algorithm_guide.md): XAttention 算法详解
- [`docs/sparse_attention_guide.md`](sparse_attention_guide.md): 稀疏 attention 方法概述
