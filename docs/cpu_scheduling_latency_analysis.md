# CPU 调度延迟分析

## 问题概述

在分析 nsys profile 时发现，chunked attention pipeline 中存在大量的 **CPU 调度延迟**，导致 GPU 利用率显著下降。

## 观察数据

### 测试环境
- GPU: NVIDIA A100-SXM4-80GB
- 模型: Llama-3.1-8B-Instruct
- 测试: RULER niah_single_1, 64K context
- Profile 文件: `ruler_8slots_test.nsys-rep`
- 时间段: 92.982s - 93.038s

### Kernel 执行时间

| Kernel | 典型执行时间 |
|--------|-------------|
| flash_fwd_kernel | ~138 μs |
| H2D memcpy (2MB) | ~87 μs |
| merge_lse_kernel | ~3.5 μs |
| merge_output_kernel | ~34 μs |

### 操作间隙分析

从 cuda_gpu_trace 观察到的间隙：

```
Start (ms)     Dur (μs)   Gap (μs)   Type
------------------------------------------------------------
92984.680      138.3      378.3      flash_fwd_kernel     ← GAP!
92985.051      86.8       232.9      H2D memcpy           ← GAP!
92985.141      86.8       2.8        H2D memcpy
92985.587      135.9      360.0      flash_fwd_kernel     ← GAP!
92986.026      3.4        302.4      merge_lse            ← GAP!
92986.164      33.5       135.0      merge_output         ← GAP!
92986.371      86.9       173.4      H2D memcpy           ← GAP!
92986.461      86.8       2.7        H2D memcpy
92986.816      137.9      268.2      flash_fwd_kernel     ← GAP!
```

### Flash Kernel 间隙分解

| 间隙 | 总时间 | 有效工作时间 | 空闲时间 |
|------|--------|-------------|---------|
| Flash 1 → Flash 2 | 769 μs | ~174 μs (2x H2D) | ~595 μs (77%) |
| Flash 2 → Flash 3 | 1092 μs | ~211 μs (merge + H2D) | ~881 μs (81%) |
| Flash 3 → Flash 4 | 965 μs | ~211 μs (merge + H2D) | ~754 μs (78%) |

**关键发现**: 每个 flash kernel 之间约 **77-81% 的时间是 CPU 调度空闲**。

## 间隙来源分析

### 1. CPU 调度延迟类型

| 转换 | 典型延迟 | 原因 |
|------|---------|------|
| Kernel 结束 → 下一个 Kernel 开始 | 100-400 μs | CPU 准备参数、调用 CUDA driver |
| Flash 结束 → H2D 开始 | ~233 μs | Python 代码执行 + CUDA launch |
| H2D 结束 → Flash 开始 | ~360 μs | 同步等待 + kernel launch |
| Flash 结束 → merge 开始 | ~302 μs | Python 代码执行 |

### 2. 延迟产生的代码位置

```python
# full_policy.py: compute_chunked_prefill

for block_idx in range(num_blocks):
    # 1. 等待 H2D 完成 (同步点)
    offload_engine.wait_slot_layer(current_slot)  # ← 可能引入延迟

    # 2. 获取 KV 数据
    k_block, v_block = offload_engine.get_kv_for_slot(current_slot)

    # 3. 调用 flash attention (kernel launch)
    block_out, block_lse = flash_attn_with_kvcache(...)  # ← CPU 调度延迟

    # 4. merge 操作
    merge_output(...)  # ← CPU 调度延迟
    merge_lse(...)     # ← CPU 调度延迟

    # 5. 发起下一个 H2D (异步)
    offload_engine.load_to_slot_layer(next_slot, ...)  # ← CPU 调度延迟
```

### 3. 为什么 H2D 之间间隙小

注意到连续的 H2D memcpy 之间间隙只有 ~2.7 μs，这是因为：
- 它们在同一个 stream 上连续发起
- CUDA driver 可以批量处理
- 没有 Python 代码介入

## GPU 利用率计算

基于观察数据：

| 指标 | 值 |
|------|-----|
| Flash kernel 平均执行时间 | 138 μs |
| Flash kernel 平均间隔 | 942 μs |
| Flash kernel GPU 利用率 | 138 / (138 + 942) = **12.8%** |

如果消除 CPU 调度延迟（仅保留必要的 H2D + merge）：

| 指标 | 值 |
|------|-----|
| 必要间隔 (2x H2D + merge) | ~211 μs |
| 理论 GPU 利用率 | 138 / (138 + 211) = **39.5%** |

**潜在提升**: 3x GPU 利用率

## 优化方向

### 1. CUDA Graph
将整个 block 处理流程编译为 CUDA Graph，消除重复的 kernel launch 开销。

```python
# 伪代码
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    # 预录制 flash + merge 操作
    block_out, block_lse = flash_attn_with_kvcache(...)
    merge_output(...)
    merge_lse(...)

# 运行时只需 replay
for block_idx in range(num_blocks):
    graph.replay()  # 单次 launch，无 Python 介入
```

### 2. 自定义 Triton Kernel
将 flash + merge 融合为单个 kernel，减少 kernel launch 次数。

### 3. C++ Extension
将 Python 循环移到 C++ 层，减少 Python 解释器开销。

### 4. 流水线重叠优化
确保 H2D 传输与前一个 block 的计算完全重叠：

```
Block 0: [H2D slot0] [Flash slot0] [merge]
Block 1:            [H2D slot1]   [Flash slot1] [merge]
Block 2:                         [H2D slot2]   [Flash slot2] [merge]
```

## 验证方法

### 1. 使用 nsys 分析间隙

```bash
# 生成 profile
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 查看 kernel trace
nsys stats --report cuda_gpu_trace --format csv <file>.nsys-rep | \
    awk -F',' 'NR>1 && $1 >= START && $1 <= END'
```

### 2. 计算间隙

```python
# 从 trace 数据计算
prev_end = start + duration
gap = next_start - prev_end
```

## 相关文件

- `nanovllm/kvcache/sparse/full_policy.py`: Pipeline 实现
- `nanovllm/kvcache/offload_engine.py`: H2D/D2H 传输
- `scripts/profile_offload.sh`: Profiling 脚本

## 参考

- [CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [nsys 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
