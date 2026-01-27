# Estimate Block Size 性能分析

本文档记录 XAttention estimate 阶段中 `block_size` 参数对 `softmax_fuse_block_sum` kernel 性能的影响。

## 问题背景

当前 `select_blocks` 中的 estimate 过程使用全局的 `kvcache_block_size`（通常为 4096）：

```python
# xattn_bsa.py: select_blocks()
block_size = ctx.block_size  # 来自 kvcache_manager.block_size (4096)
reshaped_block_size = block_size // self.stride  # 4096/8 = 512

block_sums = softmax_fuse_block_sum(
    attn_scores,
    reshaped_block_size,  # 512 - 性能最差点！
    ...
)
```

这导致 `softmax_fuse_block_sum` kernel 使用 `reshaped_block_size=512`，而这正是性能曲线的最差点。

## Benchmark 结果

### 测试配置

- GPU: NVIDIA A100-SXM4-80GB
- NUM_HEADS: 32
- HEAD_DIM: 128
- STRIDE: 8
- 测试脚本: `tests/bench_estimate_block_size.py`

### softmax_fuse_block_sum 性能数据

| block_size | reshaped | 16K context | 32K context | 64K context |
|------------|----------|-------------|-------------|-------------|
| 64 | 8 | 4.86ms | 18.36ms | 70.83ms |
| 128 | 16 | 0.83ms | 3.12ms | 16.83ms |
| 256 | 32 | 0.63ms | 2.41ms | 11.24ms |
| 512 | 64 | **0.38ms** | **1.52ms** | 9.54ms |
| 1024 | 128 | 0.42ms | 1.54ms | **6.01ms** |
| 2048 | 256 | 1.08ms | 3.24ms | 12.81ms |
| **4096** | **512** | 9.66ms | 25.36ms | **95.32ms** |

### 性能曲线

```
softmax_fuse_block_sum 耗时 (64K context):

block_size=64   ████████████████████████████████████ 70.83ms
block_size=128  ████████ 16.83ms
block_size=256  █████ 11.24ms
block_size=512  ████ 9.54ms
block_size=1024 ███ 6.01ms  ◀── 最优点
block_size=2048 ██████ 12.81ms
block_size=4096 ████████████████████████████████████████████████ 95.32ms ◀── 当前使用
```

### 关键发现

1. **性能呈 U 型曲线**：太小和太大的 block_size 都会导致性能下降
2. **最优点在 512-1024**：对应 `reshaped_block_size` 64-128
3. **当前配置 (4096) 是最差点**：95.32ms vs 最优 6.01ms，**慢 15.85x**

## 性能曲线解释

```
Performance (耗时)
    │
    │  ▲ 太小:
    │ /   - output blocks 数量多 (q_len / block_size)
    │/    - grid 调度开销大
    │     - 每个 thread block 工作量小
    │   ┌─────────┐
    │  /   最优    \
    │ /    区域     \ ▲ 太大:
    │/               \   - block_size 作为 tl.constexpr
    │                 \  - 寄存器压力增大 (可能 spill)
    │                  \ - shared memory 不足
    │                   \- L1 cache 效率下降
    └──────────────────────────────────→ block_size
      64  128  256  512 1024 2048 4096
                    ↑
              最优点 (512-1024)
```

### Triton Kernel 内部分析

`softmax_fuse_block_sum_kernel` 中的关键约束：

```python
# 每个 thread block 处理的数据
offs_q = tl.arange(0, block_size)  # block_size 个元素
m_i = tl.zeros([block_size], dtype=tl.float32)  # 寄存器分配

# reshape 操作
X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
# 当 block_size=512, segment_size=512 时 → (512, 1, 512) 的 3D tensor
```

当 `block_size` 过大时：
- 每个 thread block 需要更多寄存器
- `tl.arange(0, block_size)` 生成更大的向量
- reshape 操作的内存访问模式变差

## 优化建议

### 方案 1: 固定 estimate block_size

在 `select_blocks` 中使用固定的小 block_size 进行估计：

```python
# 建议修改
ESTIMATE_BLOCK_SIZE = 1024  # 或 512，而非 ctx.block_size

reshaped_block_size = ESTIMATE_BLOCK_SIZE // self.stride  # 128
```

**优点**：简单直接，预期提升 15x
**缺点**：estimate 的 block 粒度与 CPU block 不一致，需要映射

### 方案 2: 两级 block 结构

- 外层使用 `kvcache_block_size` (4096) 管理 CPU blocks
- 内层使用 `estimate_block_size` (1024) 进行估计
- 估计结果聚合回 CPU block 粒度

### 方案 3: 自适应 block_size

根据 context length 动态选择 estimate block_size：

| Context Length | Recommended block_size |
|----------------|------------------------|
| < 16K | 512 |
| 16K - 64K | 1024 |
| > 64K | 1024 |

## 与实际 Profiling 的对比

### Nsys Profiling 数据 (64K context, block_size=4096)

| 阶段 | 时间占比 | 说明 |
|------|----------|------|
| softmax_fuse_block_sum | **48.1%** | 最后一个 chunk |
| flash_fwd_kernel | 30.7% | 实际 attention 计算 |
| flat_group_gemm | 3.5% | estimate GEMM |

### 预期优化效果

如果将 estimate block_size 从 4096 改为 1024：

| 指标 | 当前 (4096) | 优化后 (1024) | 提升 |
|------|-------------|---------------|------|
| softmax kernel | 95.32ms | 6.01ms | **15.85x** |
| estimate 阶段占比 | 48.1% | ~5% | 显著降低 |
| 总体 prefill 时间 | ~2s (最后chunk) | ~1.1s | ~1.8x |

## 测试命令

```bash
# 运行 benchmark
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/bench_estimate_block_size.py --gpu 0

# 指定单个 context length
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/bench_estimate_block_size.py --gpu 0 --ctx-len 65536
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `nanovllm/kvcache/sparse/xattn_bsa.py` | XAttention BSA Policy 实现 |
| `nanovllm/ops/xattn.py` | Triton kernels |
| `tests/bench_estimate_block_size.py` | 性能测试脚本 |
| `docs/xattn_performance_analysis.md` | XAttention 整体性能分析 |

## 分级求和方案 (Hierarchical Block Sum)

使用小的 `estimate_block_size=1024` 计算细粒度 block_sums，然后聚合到 CPU block 级别 (4096)。

### 数学等价性

```
方案1 (block_size=4096): softmax_fuse_block_sum → [1, heads, 1, 1]
方案2 (block_size=1024): softmax_fuse_block_sum → [1, heads, 4, 4] → sum → [1, heads]

验证结果: Max difference = 0.0 ✅ 完全等价
```

### 验证代码

`tests/test_hierarchical_estimate.py` - 纯 torch + xattn kernels 实现

### 性能提升

| 指标 | 当前 (4096) | 优化后 (1024) | 提升 |
|------|-------------|---------------|------|
| softmax kernel | 12.07 ms | 0.29 ms | **41x** |
| 端到端 estimate | 95 ms | ~6 ms | **15x** |

## ⚠️ 选择策略变更

**重要**: 分级求和方案使用新的选择策略：

| 特性 | 原策略 (mask + voting) | 新策略 (score + threshold) |
|------|------------------------|----------------------------|
| 输入 | `[batch, heads, q_blocks, k_blocks]` | `[batch, heads, num_cpu_blocks]` |
| 选择粒度 | Per-q-block | Per-chunk |
| 聚合方式 | majority voting | threshold on scores |

新策略更简洁，直接利用分级求和产生的 score，避免了 mask 生成和 voting 的复杂逻辑。

## 结论

当前 estimate 阶段使用全局 `kvcache_block_size=4096` 导致 `softmax_fuse_block_sum` kernel 性能处于最差点。通过将 estimate block_size 改为 512-1024，可以获得 **15x** 的性能提升，显著降低 estimate 阶段的开销。
