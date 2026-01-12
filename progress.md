# Progress Log: nanovllm 多请求状态污染问题

## Session: 2026-01-12

### 资源分配

| 资源 | 分配 |
|------|------|
| **GPU** | **1** (严格限制，不可更改) |

### 任务目标
研究 nanovllm CPU offload 模式下多请求之间状态影响导致准确率下降的问题。

---

### 10:00 - 启动分析

**完成**:
- [x] 读取 `docs/offload_accuracy_issue.md` 了解问题背景
- [x] 激活 Serena MCP 项目
- [x] 获取关键组件符号概览

**关键文件已分析**:
- `nanovllm/kvcache/offload_engine.py` - OffloadEngine 类
- `nanovllm/kvcache/hybrid_manager.py` - HybridKVCacheManager 类
- `nanovllm/engine/model_runner.py` - ModelRunner 类
- `nanovllm/engine/llm_engine.py` - LLMEngine 类
- `nanovllm/engine/scheduler.py` - Scheduler 类

---

### 10:15 - 深入代码分析

**分析的方法**:

| 方法 | 文件 | 发现 |
|------|------|------|
| `OffloadEngine.__init__` | offload_engine.py:40-145 | 初始化所有 buffer，无 reset 方法 |
| `deallocate` | hybrid_manager.py:218-244 | 只清理逻辑块，不清理 OffloadEngine |
| `clear_decode_tracking` | hybrid_manager.py:538-549 | 清理 tracking 字典，但未被调用 |
| `run_layerwise_offload_decode` | model_runner.py:867-1057 | 包含 decode buffer 读写逻辑 |
| `generate` | llm_engine.py:114-151 | 请求循环逻辑 |
| `postprocess` | scheduler.py:93-99 | 调用 deallocate |

**关键发现 #1**: OffloadEngine 没有 reset() 方法

**关键发现 #2**: deallocate() 没有调用 clear_decode_tracking()

**关键发现 #3**: decode_buffer 在请求间不清理，可能导致状态污染

---

### 10:30 - 根因定位

**确认的问题**:

1. **decode buffer 残留**
   - 位置: `offload_engine.decode_k_buffer`, `decode_v_buffer`
   - 写入: `model_runner.py:1010-1013`
   - 读取: `model_runner.py:969-976`
   - 问题: 旧请求的 KV 数据可能被新请求读取

2. **tracking 字典未清理**
   - 位置: `hybrid_manager._decode_start_pos`, `_prefill_len`
   - 问题: 使用 `id(seq)` 作为 key，可能重用

3. **缺失的清理调用**
   - `clear_decode_tracking()` 在 `deallocate()` 中未被调用

---

### 10:45 - 创建规划文件

**创建的文件**:
- [x] `task_plan.md` - 完整的任务规划和阶段
- [x] `findings.md` - 详细的代码分析发现
- [x] `progress.md` - 本文件

---

### 11:00 - Sequential Thinking 深入分析

**使用 sequential thinking 验证分析结果**:
- 确认 deallocate() 确实没有调用 clear_decode_tracking()
- 分析 _decode_start_pos 和 _prefill_len 字典的生命周期
- 确定 id(seq) 重用是问题的触发条件

---

### 11:15 - 完成规划文件

**更新的文件**:
- [x] `task_plan.md` - 添加完整的 debug 方案和实施计划
- [x] `findings.md` - 详细的代码分析和修复方向
- [x] `progress.md` - 更新到当前进度

---

## 下一步 (待用户确认)

**执行顺序**:

1. **实施修复** - 修改 `deallocate()` 添加 `clear_decode_tracking(seq)`
2. **快速验证** - 20 样本连续执行（一次调用，不重启框架）→ 目标 20/20
3. **完整验证** - 100 样本 → 目标 100/100 (最终验收)
4. **防御性修复** (可选) - 添加 `OffloadEngine.on_sequence_finished()`

**核心修改** (一行代码):
```python
# hybrid_manager.py:deallocate() 末尾添加
self.clear_decode_tracking(seq)
```

**验收标准**:
| 测试 | 样本数 | 通过要求 |
|------|--------|----------|
| 快速验证 | 20 | 20/20 (100%) |
| 完整验证 | 100 | 100/100 (100%) |

---

## 错误记录

| 时间 | 错误 | 解决方案 |
|------|------|----------|
| 10:05 | Serena MCP 未激活 | 调用 activate_project |

---

## 文件修改记录

| 文件 | 操作 | 状态 |
|------|------|------|
| task_plan.md | 创建+更新 | 完成 |
| findings.md | 创建 | 完成 |
| progress.md | 创建+更新 | 完成 |

---

## 分析结论

**重要澄清**: nanovllm offload 模式**不支持 batch**，只能单个 request 顺序执行。问题出在**请求切换**时状态清理不完整。

**根本原因已确认**: `deallocate()` 没有调用 `clear_decode_tracking()`，导致 `_decode_start_pos` 和 `_prefill_len` 字典残留，当 Python 对象 ID 重用时，新请求会错误地使用旧请求的配置。

**修复方案已设计**: 在 `deallocate()` 末尾添加 `self.clear_decode_tracking(seq)` 调用。

---

## 关键理解

问题不是 "batch 处理"，而是：
```
Request A 完成 → deallocate(A) [状态未完全清理] → Request B 开始 → B 读到 A 的残留状态
```
