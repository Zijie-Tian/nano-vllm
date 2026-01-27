# XAttention Performance Analysis

本文档记录 XAttention 在不同配置下的性能分析结果，包括 NVTX 标记位置、block size 影响和性能瓶颈。

## NVTX 标记

XAttention 代码中添加了 NVTX 标记用于 nsys profiling，便于分析 estimate 和 compute 阶段的性能。

### 标记位置

| 模式 | 标记名称 | 文件位置 | 说明 |
|------|---------|---------|------|
| GPU-only | `xattn_estimate` | `xattn_bsa.py:compute_prefill` | xattn_estimate 调用 |
| GPU-only | `xattn_bsa_compute` | `xattn_bsa.py:compute_prefill` | BSA kernel 调用 |
| Offload | `xattn_estimate_gemm` | `xattn_bsa.py:select_blocks` | flat_group_gemm 循环 |
| Offload | `xattn_estimate_softmax` | `xattn_bsa.py:select_blocks` | softmax_fuse_block_sum |
| Offload | `xattn_estimate_find_blocks` | `xattn_bsa.py:select_blocks` | find_blocks_chunked |
| Offload | `xattn_compute_historical` | `xattn_bsa.py:compute_chunked_prefill` | 历史 chunks attention |
| Offload | `xattn_compute_current` | `xattn_bsa.py:compute_chunked_prefill` | 当前 chunk attention |
| Offload | `xattn_compute_merge` | `xattn_bsa.py:compute_chunked_prefill` | merge 操作 |

### 查看 NVTX 统计

```bash
# 生成 profile
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --block-size 4096 --gpu 0

# 查看 NVTX 统计
nsys stats --report nvtx_pushpop_sum results/nsys/<filename>.nsys-rep
```

## Block Size 对 Offload 模式的影响

### 测试配置

- Model: Llama-3.1-8B-Instruct
- Context: 64K tokens
- Mode: xattn + offload
- GPU: A100 40GB

### 性能对比

| 指标 | block_size=4096 | block_size=1024 | 变化 |
|------|----------------|-----------------|------|
| **总时间** | 27.7s | 55.5s | **2x 慢** |
| **Chunks 数量** | 16 | 64 | 4x |
| **CPU blocks** | 18 | 71 | ~4x |

### 各阶段耗时分布

#### block_size=4096

| 阶段 | 占比 | 总时间 | 平均时间 | 调用次数 |
|-----|------|--------|---------|---------|
| **xattn_estimate_find_blocks** | **39.7%** | 18.0s | 37.6ms | 480 |
| xattn_compute_historical | 4.4% | 2.0s | 4.2ms | 480 |
| xattn_estimate_gemm | 3.4% | 1.5s | 3.2ms | 480 |
| xattn_compute_current | 0.2% | 113ms | 0.22ms | 512 |
| xattn_compute_merge | 0.2% | 96ms | 0.19ms | 512 |
| xattn_estimate_softmax | 0.2% | 88ms | 0.18ms | 480 |

#### block_size=1024

| 阶段 | 占比 | 总时间 | 平均时间 | 调用次数 |
|-----|------|--------|---------|---------|
| **xattn_estimate_gemm** | **23.6%** | 22.6s | 11.4ms | 1984 |
| **xattn_compute_historical** | **16.9%** | 16.2s | 8.0ms | 2016 |
| xattn_estimate_find_blocks | 1.4% | 1.3s | 0.66ms | 1984 |
| xattn_compute_current | 0.5% | 433ms | 0.21ms | 2048 |
| xattn_compute_merge | 0.4% | 373ms | 0.18ms | 2048 |
| xattn_estimate_softmax | 0.2% | 222ms | 0.11ms | 1984 |

### 关键发现

1. **Block size 对性能影响显著**
   - block_size=1024 比 4096 慢约 2x
   - 更小的 block size 导致更多的 chunks，增加调用次数

2. **性能瓶颈随 block size 变化**
   - **block_size=4096**: 瓶颈是 `find_blocks_chunked` (39.7%)
   - **block_size=1024**: 瓶颈转移到 `estimate_gemm` (23.6%) 和 `compute_historical` (16.9%)

3. **Amortization 效应**
   - 大 block size 虽然单次 `find_blocks` 更慢 (37.6ms vs 0.66ms)
   - 但调用次数少 (480 vs 1984)，总时间反而更少

4. **find_blocks_chunked 的特殊性**
   - 该函数主要在 CPU 上执行 block 选择逻辑
   - 处理更大的数据量时开销显著增加
   - block_size=4096 时占用 40% 时间，是主要优化目标

## softmax_fuse_block_sum_kernel 性能分析

`softmax_fuse_block_sum_kernel_non_causal` 是 XAttention 估计阶段的核心 Triton kernel。

### Kernel 结构

```python
# 每个 thread block 处理的数据形状
工作负载: [block_size, segment_size]  # 单个 Q block 对所有 K 的注意力

# Pass 1: 计算全局 softmax 参数 (m_i, l_i)
for iter in range(num_iters):  # num_iters = k_len / segment_size
    X = load [block_size, segment_size]
    compute max, sum for softmax normalization

# Pass 2: Normalize + Block Sum
for iter in range(num_iters):
    X = load [block_size, segment_size]
    X = softmax(X)
    X = reshape(X, [block_size, segment_size/block_size, block_size])
    X = sum(X, axis=2)  # → [block_size, segment_size/block_size]
    X = sum(X, axis=0)  # → [segment_size/block_size]
    store output
```

### 性能随 block_size 变化的因素

| 因素 | 小 block_size (64) | 大 block_size (256) |
|------|-------------------|---------------------|
| Grid 并行度 | 高 (更多 blocks) | 低 (更少 blocks) |
| 寄存器使用 | 低 | 高 (可能 spill) |
| L2 Cache 复用 | 差 | 好 |
| 输出大小 | 大 | 小 |

### 典型性能曲线

```
Performance
    │
    │         ┌─────┐
    │        /       \
    │       /         \
    │      /           \
    │     /             \
    └────/───────────────\────────→ block_size
        64   128   256   512

最优点通常在 128-256 之间
```

## 优化建议

1. **优先使用 block_size=4096**
   - 减少 chunk 数量，降低调度开销
   - 更好的 amortization 效果

2. **优化 find_blocks_chunked**
   - 当前是 block_size=4096 的主要瓶颈
   - 考虑 GPU 加速或批量处理

3. **Pipeline 优化**
   - 利用多 slot 的 ring buffer 实现计算和传输 overlap
   - 当前已实现，但 find_blocks 是 CPU 操作，无法 overlap

## 测试命令

```bash
# GPU-only 模式 (需要 40GB+ VRAM)
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --no-offload --gpu 0

# Offload 模式，block_size=4096
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --block-size 4096 --gpu 0

# Offload 模式，block_size=1024
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --block-size 1024 --gpu 0

# 128K context
bash scripts/profile_offload.sh --policy xattn --ctx-len 128k --block-size 4096 --gpu 0
```
