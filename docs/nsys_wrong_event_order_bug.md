# Nsys "Wrong Event Order" Bug 调试记录

## 问题描述

使用 `nsys profile` 对 nanovllm 的 CPU offload 模式进行性能分析时，无法生成 `.nsys-rep` 文件，报错：

```
Importer error status: Importation failed.
Wrong event order has been detected when adding events to the collection:
new event ={ StartNs=21569539222 StopNs=21569672388 ... Type=48 }
last event ={ StartNs=22046804077 StopNs=22046805343 ... Type=48 }
```

## 环境信息

- **nsys 版本**: 2023.4.4.54-234433681190v0
- **CUDA**: 12.4
- **问题状态**: nsys 已知 bug，2024.2+ 版本已修复

## 调试过程

### 阶段 1：确定触发条件

使用 bisect 脚本 (`tests/test_nsys_bisect.py`) 逐步测试：

| Stage | 描述 | 结果 |
|-------|------|------|
| 1 | CUDA init | ✅ |
| 2 | Import nanovllm | ✅ |
| 3 | Create LLM (offload) | ✅ |
| 4 | 短 prompt 生成 | ✅ |
| **5** | **长 prompt (~64K) prefill** | ❌ |

**结论**：问题出在长 prompt 的 chunked prefill 流程。

### 阶段 2：定位具体组件

在 `_chunked_prefill_attention` 方法中逐步注释代码：

| 组件 | 文件位置 | 结果 |
|------|----------|------|
| 整个方法 (return zeros) | `attention.py:167` | ✅ |
| `select_blocks()` | `attention.py:217` | ✅ |
| `offload_prefill_buffer_async()` | `attention.py:241-248` | ✅ |
| `compute_chunked_prefill()` | `attention.py:225-235` | ❌ |

**结论**：问题出在 `compute_chunked_prefill` 内部。

### 阶段 3：定位 Ring Buffer Pipeline

在 `full_policy.py` 中进一步定位：

| 组件 | 代码行 | 结果 |
|------|--------|------|
| Current chunk attention | 191-198 | ✅ |
| **Historical block loading (ring buffer)** | 133-189 | ❌ |

**根因确认**：Ring buffer pipeline 的多 stream 操作触发了 nsys bug。

## 根本原因

### 触发 Bug 的代码

```python
# nanovllm/kvcache/sparse/full_policy.py:133-189

# 多 slot pipeline 模式
for block_idx in range(num_blocks):
    current_slot = load_slots[block_idx % num_slots]

    # 等待 slot 的 transfer stream 完成
    offload_engine.wait_slot_layer(current_slot)

    # 在 compute_stream 上执行 attention
    with torch.cuda.stream(compute_stream):
        prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
        prev_o, prev_lse = flash_attn_with_lse(...)
        offload_engine.record_slot_compute_done(current_slot)

    # 异步发起下一个 block 的加载
    if next_block_idx < num_blocks:
        offload_engine.load_to_slot_layer(next_slot, layer_id, next_cpu_block_id)
```

### Stream 结构

```
slot_transfer_streams[0] ─┐
slot_transfer_streams[1] ─┼─ 4 个 transfer streams
slot_transfer_streams[2] ─┤
slot_transfer_streams[3] ─┘
                          │
                          ▼ wait/record 同步
                          │
compute_stream ───────────┘
```

这种 4+1 stream 的复杂同步模式导致 nsys 2023.4.4 版本的事件时间戳排序算法出错。

### 为什么简单多 stream 测试无法复现

我们尝试用简单的测试代码 (`tests/test_multistream_nsys.py`) 复现问题：

- 4-8 streams, 2000+ iterations: ✅ 成功
- 32 threads + multi-stream: ✅ 成功
- >64k CUDA operations: ✅ 成功

但都无法触发 bug。原因是实际代码中的 stream 同步模式更复杂：
1. 跨 stream 的 event wait/record
2. 与 FlashAttention kernel 的交互
3. 长时间运行（~50 秒）累积大量事件

## 解决方案

### 方案 1：升级 nsys（推荐）

```bash
# 下载 nsys 2024.2+ 版本
# https://developer.nvidia.com/nsight-systems
```

根据 [NVIDIA 论坛](https://forums.developer.nvidia.com/t/nsys-profiler-wrong-event-order/264881)，此 bug 在 2024.2 版本已修复。

### 方案 2：使用 .qdstrm 文件

即使导入失败，`.qdstrm` 文件仍然生成：

```bash
# 生成的文件
results/nsys/ruler_niah_single_1_sample0_offload_*.qdstrm

# 尝试用 GUI 直接打开
nsight-sys <file>.qdstrm
```

GUI 可能有更好的容错能力。

### 方案 3：使用 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # your code

prof.export_chrome_trace("trace.json")  # chrome://tracing 查看
```

### 方案 4：临时禁用 ring buffer pipeline

在 `full_policy.py` 中临时使用单 slot 同步模式（仅用于调试）：

```python
# 强制使用单 slot 模式
if len(load_slots) == 1 or True:  # 添加 "or True"
    # 同步模式，不会触发 nsys bug
    ...
```

## 复现步骤

### 环境准备

```bash
cd /home/zijie/Code/nano-vllm
```

### 运行 Bisect 脚本

```bash
# Stage 5 会触发 bug
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PYTHONPATH \
    nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    -o /tmp/bisect python tests/test_nsys_bisect.py --stage 5
```

### 验证修复

```bash
# 临时在 full_policy.py 中跳过 historical block loading
# 将第 133 行改为: if False and cpu_block_table:

# 重新运行，应该成功
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PYTHONPATH \
    nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    -o /tmp/bisect_fixed python tests/test_nsys_bisect.py --stage 5

# 检查是否生成 .nsys-rep
ls -la /tmp/bisect_fixed.nsys-rep
```

## 相关文件

| 文件 | 用途 |
|------|------|
| `tests/test_nsys_bisect.py` | Bisect 调试脚本 |
| `tests/test_multistream_nsys.py` | 简单多 stream 测试 |
| `scripts/profile_offload.sh` | nsys profile 脚本 |
| `nanovllm/layers/attention.py` | Attention 层 |
| `nanovllm/kvcache/sparse/full_policy.py` | Ring buffer pipeline |

## 参考资料

- [Nsys Profiler- Wrong event order - NVIDIA Forums](https://forums.developer.nvidia.com/t/nsys-profiler-wrong-event-order/264881)
- [Nsight Systems 2025.3 Release Notes](https://docs.nvidia.com/nsight-systems/2025.3/ReleaseNotes/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 调试日期

2026-01-24
