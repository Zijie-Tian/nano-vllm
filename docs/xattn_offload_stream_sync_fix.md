# XAttention Offload Stream Synchronization Fix

修复 XAttention BSA Policy 在 Offload 模式下的 CUDA stream 同步 bug。

**修复日期**: 2026-02-05
**Commit**: `829b311`
**影响文件**: `nanovllm/kvcache/sparse/xattn_bsa.py`, `nanovllm/kvcache/offload_engine.py`

---

## 问题描述

### 症状

在 Offload 模式下运行 RULER benchmark 时，XAttention BSA 的 `select_blocks` 方法中 Pass 1 和 Pass 2 从**同一个 CPU block** 加载的 K 数据不一致：

```
Pass 1: K_chunk sum = 745472.00 (正确)
Pass 2: K_chunk sum = 0.00      (错误，数据未加载完成)
```

这导致 attention 计算结果错误，RULER 准确率下降。

### 复现条件

- 模式: Offload (`--enable-offload`)
- Context: ≥ 32K tokens
- 稀疏策略: `--sparse-policy XATTN_BSA`

---

## 根因分析

### Stream 配置回顾

nano-vllm 的 CPU offload 使用多个 CUDA streams 实现 pipeline：

| Stream | 用途 |
|--------|------|
| `slot_transfer_streams[i]` | H2D 传输 (CPU → GPU slot) |
| `compute_stream` | Attention 计算 |
| `prefill_offload_streams[i]` | D2H 传输 (GPU → CPU cache) |

### 同步机制

`wait_slot_layer(slot)` 使用 event 机制同步：

```python
def wait_slot_layer(self, slot_idx: int):
    """Make compute_stream wait for H2D transfer completion."""
    self.compute_stream.wait_event(self.ring_slot_ready[slot_idx])
```

### Bug 根因

在 `select_blocks` 方法中：

1. H2D 传输在 `slot_transfer_streams` 上执行
2. `wait_slot_layer` 让 `compute_stream` 等待传输完成
3. **但是** 后续的 compute kernels 在**默认 stream** 上执行，而不是 `compute_stream`

```python
# Bug 代码
offload_engine.load_k_only_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)  # compute_stream 等待

# 这些 kernel 在默认 stream 上运行，没有等待 H2D 完成！
k_block = offload_engine.get_k_for_slot(slot)
K_chunk = k_block.transpose(1, 2)
# ... 后续计算 ...
```

### 时序图

```
slot_transfer_stream:  [====H2D====]
compute_stream:             |wait|
default_stream:        [kernel1][kernel2]  ← 没有等待！
                            ↑
                       数据未就绪
```

---

## 修复方案

### 核心修改

将所有 estimate 阶段的 compute kernels 包装在 `with torch.cuda.stream(compute_stream):` 中：

```python
# 修复后代码
compute_stream = offload_engine.compute_stream

offload_engine.load_k_only_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)  # compute_stream 等待

# 所有计算在 compute_stream 上执行
with torch.cuda.stream(compute_stream):
    k_block = offload_engine.get_k_for_slot(slot)
    K_chunk = k_block.transpose(1, 2)
    # ... 后续计算 ...
```

### 修复位置

`select_blocks` 方法中共 6 处需要修复：

| 位置 | 阶段 | 修复内容 |
|------|------|----------|
| Pass 1 历史 blocks | `xattn_estimate_pass1` | 历史 KV chunk 处理 |
| Pass 1 当前 chunk | `xattn_estimate_pass1` | 当前 GPU 上的 K 处理 |
| Step 2 合并 | `merge_softmax_stats` | softmax stats 合并 |
| Pass 2 历史 blocks | `xattn_estimate_pass2` | 带全局 stats 的 block_sum |
| Pass 2 当前 chunk | `xattn_estimate_pass2` | 当前 chunk 的 block_sum |
| Step 4 block 选择 | `find_blocks_chunked` | 最终 block 选择 |

### 时序图（修复后）

```
slot_transfer_stream:  [====H2D====]
compute_stream:             |wait|[kernel1][kernel2]
                                   ↑
                              数据已就绪
```

---

## 代码变更详情

### 1. Pass 1 历史 blocks 处理

```python
# Before (bug)
for kv_chunk_idx, cpu_block_id in enumerate(available_blocks):
    offload_engine.load_k_only_to_slot_layer(slot, layer_id, cpu_block_id)
    offload_engine.wait_slot_layer(slot)

    k_block = offload_engine.get_k_for_slot(slot)  # 默认 stream
    K_chunk = k_block.transpose(1, 2)
    # ... compute ...

# After (fixed)
compute_stream = offload_engine.compute_stream

for kv_chunk_idx, cpu_block_id in enumerate(available_blocks):
    offload_engine.load_k_only_to_slot_layer(slot, layer_id, cpu_block_id)
    offload_engine.wait_slot_layer(slot)

    with torch.cuda.stream(compute_stream):  # 显式指定 stream
        k_block = offload_engine.get_k_for_slot(slot)
        K_chunk = k_block.transpose(1, 2)
        # ... compute ...
```

### 2. 移除 STRONG SYNC

`offload_engine.py` 中移除了不必要的强同步：

```python
# Removed from load_to_slot_layer() and load_k_only_to_slot_layer()
# STRONG SYNC: Synchronize all prefill offload streams before H2D
# for offload_stream in self.prefill_offload_streams:
#     offload_stream.synchronize()
```

这些同步现在由 event 机制正确处理，不再需要阻塞式同步。

### 3. 其他清理

- 移除 DEBUG print 语句
- 移除 `torch.save()` debug 代码
- 合并多个 fallback 条件
- 将 `chunk_size` 默认值从 16384 改为 4096（匹配 offload Q chunk size）

---

## 测试验证

### 测试命令

**GPU 0 - Offload 模式测试**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 10 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

**GPU 1 - GPU-only 模式测试**:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Qwen3-0.6B \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 10 \
    --max-model-len 40960 \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.9
```

### 测试结果

| 模式 | 模型 | Context | Samples | Pass Rate | Density |
|------|------|---------|---------|-----------|---------|
| Offload | Llama-3.1-8B | 32K | 10/10 | **100%** | 9.53% |
| GPU-only | Qwen3-0.6B | 32K | 10/10 | **100%** | 9.84% |

### Density 对齐验证

| 模式 | Layer 0 Density | 差异 |
|------|-----------------|------|
| GPU-only | 9.84% | - |
| Offload | 9.53% | ~3% |

~3% 的差异是预期的，因为两种模式的 KV 累积模式不同：
- GPU-only: 一次性处理所有 KV
- Offload: 分 chunk 处理，每个 chunk 独立计算 softmax stats 后合并

---

## 技术细节

### 三阶段 KV Chunking 流程

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: softmax_compute_partial_stats                     │
│    └── 每个 KV chunk 独立计算 partial stats (m_i, l_i)       │
│                                                             │
│  Stage 2: merge_softmax_stats                               │
│    └── Host 端合并所有 chunks: (m_global, l_global)          │
│                                                             │
│  Stage 3: softmax_normalize_and_block_sum                   │
│    └── 使用全局 stats 归一化并计算 block sums                 │
└─────────────────────────────────────────────────────────────┘
```

### Stream 配置要求

| 操作类型 | Stream | 原因 |
|----------|--------|------|
| H2D 传输 | `slot_transfer_streams` | 异步传输，不阻塞计算 |
| D2H 传输 | `prefill_offload_streams` | 异步 offload，不阻塞计算 |
| Estimate kernels | `compute_stream` | 与 attention 计算共享，确保同步 |
| Attention kernels | `compute_stream` | 主计算流 |

### Event 同步机制

```python
# H2D 传输完成后记录 event
self.ring_slot_ready[slot_idx].record(slot_transfer_stream)

# 计算前等待 H2D 完成
self.compute_stream.wait_event(self.ring_slot_ready[slot_idx])

# 计算完成后记录 event（用于下一轮 H2D）
self.ring_slot_compute_done[slot_idx].record(compute_stream)
```

---

## 相关文档

- [`docs/architecture_guide.md`](architecture_guide.md): Stream 配置和 ring buffer 架构
- [`docs/xattn_kv_chunking_kernels.md`](xattn_kv_chunking_kernels.md): 三阶段 softmax kernels
- [`docs/gpuonly_density_alignment_test.md`](gpuonly_density_alignment_test.md): Density 对齐测试
- [`docs/xattn_bsa_policy_design.md`](xattn_bsa_policy_design.md): XAttention BSA Policy 设计

---

## 经验总结

### 1. Stream 同步的隐蔽性

CUDA stream 同步 bug 很难发现：
- 数据可能"大部分时间"正确（取决于时序）
- 错误表现为随机/间歇性的结果偏差
- 需要精确的 debug logging 才能定位

### 2. Event vs Synchronize

| 方法 | 优点 | 缺点 |
|------|------|------|
| `stream.wait_event()` | 非阻塞，保持 pipeline | 只同步指定 stream |
| `stream.synchronize()` | 保证完成 | 阻塞整个 stream，破坏 pipeline |

**最佳实践**: 使用 event 进行精确同步，避免 synchronize 阻塞。

### 3. 调试方法

```python
# 打印 tensor sum 验证数据一致性
print(f"K_chunk sum = {K_chunk.sum().item()}")

# 保存中间结果进行离线比较
torch.save({'K': K_chunk, 'layer': layer_id}, f'/tmp/debug_{pass}_{chunk}.pt')
```
