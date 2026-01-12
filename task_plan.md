# Task Plan: nanovllm CPU Offload 多请求状态污染问题

## 问题概述

**重要说明**: nanovllm offload 模式目前**不支持 batch**，只能单个 request 顺序执行。问题出在**请求切换**时的状态清理。

| 模式 | 测试方式 | 准确率 |
|------|----------|--------|
| CPU Offload | 独立进程 (每请求一个进程) | **100%** |
| CPU Offload | 同进程顺序多请求 | 66% |
| Non-Offload | 同进程顺序多请求 | 100% |

**结论**: 单请求推理正确，问题在于**请求切换**时状态清理不完整。

---

## Phase 1: 代码分析 (complete)

### 1.1 识别状态管理组件

**已分析的关键组件**:

| 组件 | 文件 | 状态数据 |
|------|------|----------|
| `OffloadEngine` | `nanovllm/kvcache/offload_engine.py` | ring buffer, decode buffer, CUDA events |
| `HybridKVCacheManager` | `nanovllm/kvcache/hybrid_manager.py` | logical blocks, prefilled_blocks, _decode_start_pos, _prefill_len |
| `LLMEngine` | `nanovllm/engine/llm_engine.py` | generate() 循环，请求生命周期 |
| `Scheduler` | `nanovllm/engine/scheduler.py` | postprocess() 调用 deallocate() |

### 1.2 请求生命周期分析

```
generate()
  → 多个请求添加到 scheduler
  → while not finished:
      → schedule() 获取下一批 seqs
      → model_runner.run() 执行推理
      → postprocess() 处理完成的请求
          → 如果完成: kvcache_manager.deallocate(seq)
```

---

## Phase 2: 根本原因分析 (complete)

### 2.1 核心问题: OffloadEngine 缺少 reset() 方法

**关键发现**: `OffloadEngine` 没有任何重置/清理方法！

当请求完成时，`HybridKVCacheManager.deallocate()` 被调用，但它只清理：
- 逻辑块状态 (`block.reset()`)
- 物理块引用 (`free_cpu_blocks`, `cpu_block_to_logical`)
- prefilled_blocks 集合
- _decode_start_pos / _prefill_len 字典

**未被清理的状态** (存在于 OffloadEngine):

| 状态 | Shape | 问题 |
|------|-------|------|
| `layer_k_cache` | [num_buffers, max_seq_len, kv_heads, head_dim] | 包含旧请求的 KV |
| `layer_v_cache` | [num_buffers, max_seq_len, kv_heads, head_dim] | 包含旧请求的 KV |
| `decode_k_buffer` | [num_layers, block_size, kv_heads, head_dim] | 包含旧请求的 decode KV |
| `decode_v_buffer` | [num_layers, block_size, kv_heads, head_dim] | 包含旧请求的 decode KV |

### 2.2 具体污染场景

在 `run_layerwise_offload_decode()` (model_runner.py:867-1057):

```python
# 第 969-976 行: 读取之前的 decode KV
if num_prev_decode_tokens > 0:
    k_decode_prev, v_decode_prev = offload_engine.get_decode_kv(
        layer_id, decode_start_pos, pos_in_block
    )
    ring_k[...].copy_(k_decode_prev)  # 可能读取旧请求的数据!
```

**场景**:
1. 请求 A (32K tokens) 完成，decode_buffer 保留其 KV 数据
2. 请求 B 开始，其 `decode_start_pos` 可能非零（如果继承了旧状态）
3. 请求 B 在第一个 decode step 时错误地读取了请求 A 的 decode buffer 数据

### 2.3 潜在问题点

1. **decode_start_pos 计算错误**:
   - `get_decode_start_pos()` 使用 `id(seq)` 作为 key
   - Python 对象 ID 可能在请求之间重用
   - 如果新 seq 对象的 ID 与旧 seq 相同，可能错误继承旧的 start_pos

2. **decode buffer 残留数据**:
   - 如果 `pos_in_block` 在新请求中与旧请求重叠
   - `get_decode_kv()` 会返回旧请求的数据

3. **ring buffer 残留数据**:
   - 虽然每次 decode 会从 CPU 加载，但 decode buffer 的数据会被复制过来
   - 如果 decode buffer 有残留，会污染 ring buffer

---

## Phase 3: Debug 方案设计 (complete)

### 3.1 确认的根本原因

通过代码分析，确认了两个根本原因：

**根本原因 1 (主要)**: `deallocate()` 不调用 `clear_decode_tracking()`
- 位置: `hybrid_manager.py:218-244`
- 影响: `_decode_start_pos` 和 `_prefill_len` 字典残留
- 后果: 如果 `id(seq)` 重用，返回错误的 decode 配置

**根本原因 2 (次要)**: decode_buffer 不清理
- 位置: `offload_engine.py`
- 影响: `decode_k_buffer/v_buffer` 保留旧 KV
- 后果: 可能被根本原因 1 触发读取

### 3.2 Debug 方案 A: 验证字典残留 (推荐先做)

**目标**: 验证 `_decode_start_pos` 字典是否有残留

**诊断代码** (添加到 `hybrid_manager.py`):
```python
# 在 get_decode_start_pos() 开头添加
def get_decode_start_pos(self, seq: Sequence) -> int:
    seq_id = id(seq)
    # DEBUG: 检查是否命中旧值
    if seq_id in self._decode_start_pos:
        logger.warning(f"[DEBUG] get_decode_start_pos: CACHE HIT! seq_id={seq_id}, "
                       f"cached_value={self._decode_start_pos[seq_id]}, "
                       f"expected={(len(seq) - 1) % self._block_size}")
    # ... 原有逻辑
```

**诊断代码** (添加到 `deallocate()` 末尾):
```python
def deallocate(self, seq: Sequence) -> None:
    # ... 现有逻辑 ...

    # DEBUG: 打印未清理的状态
    seq_id = id(seq)
    if seq_id in self._decode_start_pos:
        logger.warning(f"[DEBUG] deallocate: _decode_start_pos NOT CLEARED! "
                       f"seq_id={seq_id}, value={self._decode_start_pos[seq_id]}")
```

### 3.3 Debug 方案 B: 最小复现测试

**文件**: `tests/test_multi_request_offload_debug.py`

```python
"""最小复现批量模式失败"""
import os
import sys
sys.path.insert(0, os.getcwd())

from nanovllm import LLM
from nanovllm.sampling import SamplingParams

# 使用 RULER NIAH 的两个样本
PROMPTS = [
    # Sample 0 (通常成功)
    "...",  # 从 niah_single_1_32k.jsonl 加载
    # Sample 1 (通常失败)
    "...",
]
EXPECTED = ["8930103", "4194548"]

def main():
    llm = LLM(
        "~/models/Llama-3.1-8B-Instruct",
        max_model_len=33792,
        max_num_batched_tokens=33792,
        enable_cpu_offload=True,
        num_gpu_blocks=4,
        kvcache_block_size=1024,
        enforce_eager=True,
    )

    params = SamplingParams(temperature=0.1, max_tokens=50)

    # 连续处理两个请求
    for i, (prompt, expected) in enumerate(zip(PROMPTS, EXPECTED)):
        print(f"\n{'='*60}")
        print(f"Sample {i}: Expected = {expected}")

        # 打印关键状态
        kvm = llm.model_runner.kvcache_manager
        print(f"  _decode_start_pos 字典大小: {len(kvm._decode_start_pos)}")
        print(f"  _prefill_len 字典大小: {len(kvm._prefill_len)}")

        outputs = llm.generate([prompt], params, use_tqdm=False)
        output_text = outputs[0]["text"]

        passed = expected in output_text
        print(f"  Output: {output_text[:100]}...")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main()
```

### 3.4 Debug 方案 C: 快速修复验证

**目标**: 验证修复 `deallocate()` 是否解决问题

**修改** (`hybrid_manager.py:218-244`):
```python
def deallocate(self, seq: Sequence) -> None:
    """Release all blocks for a sequence."""
    for logical_id in reversed(seq.block_table):
        # ... 现有逻辑 ...

    seq.num_cached_tokens = 0
    seq.block_table.clear()

    # === 新增: 清理 decode tracking ===
    self.clear_decode_tracking(seq)
```

**验证命令**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sample-indices 0,1,2,3,4 \
    --verbose
```

### 3.5 Debug 方案 D: 添加 OffloadEngine 清理 (防御性)

**目标**: 进一步隔离请求状态

**添加方法** (`offload_engine.py`):
```python
def on_sequence_finished(self):
    """清理请求完成后的状态"""
    # 清零 decode buffer (防止残留数据被读取)
    self.decode_k_buffer.zero_()
    self.decode_v_buffer.zero_()
    logger.debug("OffloadEngine: decode buffer cleared")
```

**调用点** (`hybrid_manager.py:deallocate` 末尾):
```python
# 清理 OffloadEngine 状态
if self.offload_engine is not None:
    self.offload_engine.on_sequence_finished()
```

---

## Phase 4: 实施计划 (pending)

### 推荐执行顺序

1. **Step 4.1**: 实施修复
   - 修改 `hybrid_manager.py:deallocate()` 添加 `clear_decode_tracking(seq)`

2. **Step 4.2**: 快速验证 (20 样本连续执行)
   - **一次调用** `test_ruler_niah.py`，连续执行 20 个样本
   - **不重启框架**，验证请求切换是否正确
   - 目标: 20/20 全部通过

3. **Step 4.3**: 完整验证 (100 样本)
   - 运行 100 个样本的 RULER NIAH 测试
   - 目标: 100/100 全部通过 (准确率从 66% → 100%)

4. **Step 4.4**: 防御性修复 (可选)
   - 添加 `OffloadEngine.on_sequence_finished()` 方法
   - 清零 decode buffer 作为额外保险

### 具体修改

**文件 1**: `nanovllm/kvcache/hybrid_manager.py`

位置: `deallocate()` 方法末尾 (第 244 行后)

```python
def deallocate(self, seq: Sequence) -> None:
    """Release all blocks for a sequence."""
    for logical_id in reversed(seq.block_table):
        # ... 现有逻辑 (218-242 行) ...

    seq.num_cached_tokens = 0
    seq.block_table.clear()

    # ============ 新增: 清理 decode tracking ============
    self.clear_decode_tracking(seq)
```

**文件 2** (可选): `nanovllm/kvcache/offload_engine.py`

位置: 在类末尾添加新方法

```python
def on_sequence_finished(self):
    """清理请求完成后的状态 (防御性清理)"""
    self.decode_k_buffer.zero_()
    self.decode_v_buffer.zero_()
```

---

## 关键文件清单

| 文件 | 相关行号 | 说明 |
|------|----------|------|
| `nanovllm/kvcache/hybrid_manager.py` | 218-244 | `deallocate()` - **需要修改** |
| `nanovllm/kvcache/hybrid_manager.py` | 538-549 | `clear_decode_tracking()` - 已存在 |
| `nanovllm/kvcache/hybrid_manager.py` | 485-505 | `get_decode_start_pos()` - 问题读取点 |
| `nanovllm/kvcache/hybrid_manager.py` | 519-537 | `get_prefill_len()` - 问题读取点 |
| `nanovllm/kvcache/offload_engine.py` | 40-145 | `__init__` - 状态初始化 |
| `nanovllm/kvcache/offload_engine.py` | (新增) | `on_sequence_finished()` - 可选防御 |
| `nanovllm/engine/model_runner.py` | 867-1057 | `run_layerwise_offload_decode()` |
| `nanovllm/engine/model_runner.py` | 969-976 | decode buffer 读取 (污染点) |

---

## 验证命令

**指定 GPU: 1** (严格限制，不可更改)

```bash
# 快速验证 (20 样本连续执行，不重启框架)
# 目标: 20/20 通过
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sample-indices 0-19 \
    --verbose

# 完整验证 (100 样本)
# 目标: 100/100 通过 (最终验收)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --quiet
```

**验收标准**:
| 测试 | 样本数 | 通过要求 | 说明 |
|------|--------|----------|------|
| 快速验证 | 20 | 20/20 (100%) | 一次调用，连续执行，验证请求切换 |
| 完整验证 | 100 | 100/100 (100%) | 最终验收 |

---

## 当前状态

- [x] Phase 1: 代码分析
- [x] Phase 2: 根本原因分析
- [x] Phase 3: Debug 方案设计
- [x] Phase 4: 实施计划 ✅ 100/100 PASSED

### 验证结果

| 测试 | 结果 | 日期 |
|------|------|------|
| 20 样本快速验证 | ✅ 20/20 (100%) | 2026-01-13 |
| 100 样本完整验证 | ✅ 100/100 (100%) | 2026-01-13 |
