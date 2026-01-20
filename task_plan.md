# nanovllm 状态泄漏调试计划

**Task**: 修复连续请求之间的状态泄漏，使 RULER 32K 测试准确率从 ~80% 提升到 100%
**Created**: 2026-01-20
**Updated**: 2026-01-21
**Status**: `in_progress`
**Reference**: [docs/ruler_32k_chunked_offload_issue.md](docs/ruler_32k_chunked_offload_issue.md)

---

## Root Cause Summary

**已确认问题**: 连续请求之间的状态泄漏
- **证据**: 单样本测试（每次重新初始化 LLM）准确率 100%，批量测试准确率 ~80%
- **差异**: 20% 的错误率来自状态泄漏

---

## 调试规范 (MUST FOLLOW)

### 1. 并行工作流（多 GPU + 异步 Task + 标志文件）

```
┌─────────────────────────────────────────────────────────────┐
│  GPU 0: 异步 Task - 当前版本测试 (v1)                        │
│    → 结果: /tmp/nanovllm_test_v1.json                       │
│    → 标志: /tmp/nanovllm_test_v1.done                       │
├─────────────────────────────────────────────────────────────┤
│  GPU 1: 主 Agent - 调试/修复代码                             │
│    → 状态对比验证                                            │
│    → 修改代码                                                │
├─────────────────────────────────────────────────────────────┤
│  GPU 2: 异步 Task - 新版本测试 (v2, 修复后)                  │
│    → 结果: /tmp/nanovllm_test_v2.json                       │
│    → 标志: /tmp/nanovllm_test_v2.done                       │
├─────────────────────────────────────────────────────────────┤
│  主 Agent: 检查标志文件，读取结果，决定下一步                 │
└─────────────────────────────────────────────────────────────┘
```

**标志文件约定**：
- 结果文件: `/tmp/nanovllm_test_<version>.json`
- 完成标志: `/tmp/nanovllm_test_<version>.done`

### 2. 验证测试命令

```bash
# 批量测试 niah_single_1 (100 samples) - 作为验证手段
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py --task niah_single_1 --enable-offload --json-output /tmp/result.json

# 完成后写标志文件
echo "done" > /tmp/nanovllm_test_done.flag
```

### 3. Phase 完成后报告

每个 Phase 完成后：
1. **更新 `progress.md`** - 记录测试结果和发现
2. **向用户报告** - 总结本 phase 的结果
3. **等待用户确认** - 不要自动开始下一个 phase

---

## 问题优先级（已更新）

| 优先级 | 问题 | 文件位置 | 状态 |
|--------|------|----------|------|
| **P0** | CPU cache 未清除 | `offload_engine.py:reset()` | 需要验证 |
| **P0** | Ring buffer slot 状态 | `offload_engine.py` | 需要验证 |
| **P1** | Sparse policy 状态 | `sparse/policy.py` | 待检查 |
| **P2** | HybridManager 跟踪变量 | `hybrid_manager.py` | 待检查 |

---

## Phase 0: 状态分析

**Status**: `completed`
**Objective**: 分析代码中的状态管理逻辑

### 发现

#### OffloadEngine.reset() 分析
**文件**: `nanovllm/kvcache/offload_engine.py:247-274`

| 组件 | reset() 是否清除 |
|------|-----------------|
| GPU ring buffer (k/v_cache_gpu) | Yes |
| Decode buffers (decode_k/v_buffer) | Yes |
| Prefill buffers (prefill_k/v_buffer) | Yes |
| Pending events | Yes |
| **CPU cache (k/v_cache_cpu)** | **No** |
| Ring buffer slot 状态 | 需要验证 |

#### HybridKVCacheManager.deallocate() 分析
**文件**: `nanovllm/kvcache/hybrid_manager.py:206-237`

- 释放 logical blocks
- 释放 CPU blocks
- 调用 `offload_engine.reset()`
- 只在 sequence 完成时调用

#### LLMEngine.generate() 分析
**文件**: `nanovllm/engine/llm_engine.py:84-142`

- 调用 `Observer.complete_reset()` 重置性能观察器
- **没有显式调用 KV cache 重置**

---

## Phase 1: 状态一致性验证

**Status**: `in_progress`
**Objective**: 对比 fresh-llm 模式和 batch 模式下的初始状态，找出差异

### 验证思路

```
fresh-llm 模式: 每个 request 新建 LLM → 状态必定干净 → 100% 准确
batch 模式:     复用 LLM 实例 → 状态可能残留 → ~80% 准确

差异 = 泄漏源
```

### Tasks

- [ ] 1.1 添加状态 dump 函数到代码中
- [ ] 1.2 运行 fresh-llm 模式，记录某个 sample (如 #40) 开始时的状态
- [ ] 1.3 运行 batch 模式，记录同一个 sample 开始时的状态
- [ ] 1.4 对比两个状态，找出差异项

### 需要检查的状态

| 组件 | 状态 | fresh-llm | batch | 差异? |
|------|------|-----------|-------|-------|
| OffloadEngine | k_cache_cpu.sum() | - | - | - |
| OffloadEngine | v_cache_cpu.sum() | - | - | - |
| OffloadEngine | k_cache_gpu.sum() | - | - | - |
| OffloadEngine | v_cache_gpu.sum() | - | - | - |
| OffloadEngine | decode_k_buffer.sum() | - | - | - |
| OffloadEngine | prefill_k_buffer.sum() | - | - | - |
| HybridManager | len(prefilled_blocks) | - | - | - |
| HybridManager | len(free_logical_ids) | - | - | - |

### 预期结果

找到具体哪些状态在 batch 模式下不为零（或不同），这些就是泄漏源。

---

## Phase 2: 修复泄漏源

**Status**: `pending`
**Objective**: 根据 Phase 1 的发现，修复具体的泄漏点

### Tasks

- [ ] 2.1 根据 Phase 1 确定的差异项，添加对应的清除逻辑
- [ ] 2.2 运行验证测试

---

## Phase 3: 验证修复效果

**Status**: `pending`
**Objective**: 确认修复后准确率达到 100%

### Tasks

- [ ] 3.1 运行 batch 模式测试 (niah_single_1)
- [ ] 3.2 对比修复前后准确率
- [ ] 3.3 运行其他 task (multikey) 验证

### Target

| Task | 修复前 | 修复后目标 |
|------|--------|-----------|
| niah_single_1 (batch) | ~80% | 100% |
| niah_single_1 (fresh-llm) | 100% | 100% (baseline) |
| multikey_1 | ~94% | 100% |
| multikey_2 | ~94% | 100% |
| multikey_3 | ~56% | >90% |

---

## Errors Encountered

| Error | Phase | Attempt | Resolution |
|-------|-------|---------|------------|
| (待记录) | - | - | - |

---

## Decision Log

| Decision | Reason | Phase |
|----------|--------|-------|
| 使用状态一致性对比验证 | 直接对比差异，不需要逐个猜测泄漏源 | 1 |
| 使用 fresh-llm 作为 baseline | 确认单样本测试 100% 通过 | 0 |

---

## Files to Modify

| File | Modification | Phase |
|------|--------------|-------|
| `nanovllm/kvcache/offload_engine.py` | 在 reset() 添加 CPU cache 清零 | 1 |
| `nanovllm/kvcache/offload_engine.py` | 添加 slot 状态重置 | 2 |
| `nanovllm/kvcache/sparse/policy.py` | 添加 reset() 如需要 | 3 |

---

## References

- [docs/ruler_32k_chunked_offload_issue.md](docs/ruler_32k_chunked_offload_issue.md) - 问题背景
- [docs/architecture_guide.md](docs/architecture_guide.md) - 架构参考
- [findings.md](findings.md) - 代码分析发现
- [progress.md](progress.md) - 进度日志
