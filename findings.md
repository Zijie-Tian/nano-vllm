# Findings: nanovllm State Leakage Investigation

## Key Discovery 1: OffloadEngine.reset() 不清除 CPU Cache
**File**: `nanovllm/kvcache/offload_engine.py:247-274`

```python
def reset(self) -> None:
    # 清除 GPU ring buffer slots
    self.k_cache_gpu.zero_()
    self.v_cache_gpu.zero_()
    # 清除 per-layer decode buffers
    self.decode_k_buffer.zero_()
    self.decode_v_buffer.zero_()
    # 清除 per-layer prefill buffers
    self.prefill_k_buffer.zero_()
    self.prefill_v_buffer.zero_()
    # 清除 pending async events
    self.pending_events.clear()

    # ⚠️ 注意：以下内容未被清除！
    # - self.k_cache_cpu
    # - self.v_cache_cpu
    # - Ring buffer slot states
```

**Impact**: CPU cache 在请求之间保留，可能导致状态泄漏。

## Key Discovery 2: deallocate() 调用 reset()
**File**: `nanovllm/kvcache/hybrid_manager.py:206-237`

`HybridKVCacheManager.deallocate()` 方法：
1. 释放所有 logical blocks
2. 释放对应的 CPU blocks
3. **调用 `offload_engine.reset()`**

但这只在 sequence 完成被释放时发生。如果 deallocate 没有被正确调用，或者调用后 CPU cache 仍有残留数据，就会导致状态泄漏。

## Key Discovery 3: LLMEngine 没有显式重置 KV cache
**File**: `nanovllm/engine/llm_engine.py:84-142`

`LLMEngine.generate()` 方法：
- 调用 `Observer.complete_reset()` 重置性能观察器
- **没有调用任何 KV cache 重置方法**

这意味着如果前一个请求的状态没有被完全清理，会影响下一个请求。

## Key Discovery 4: 状态跟踪变量
**File**: `nanovllm/kvcache/hybrid_manager.py`

HybridKVCacheManager 维护多个状态跟踪变量：
- `prefilled_blocks: Set[int]` - 跟踪已 prefill 的 blocks
- `_decode_start_pos: Dict[int, int]` - 每个 sequence 的 decode 起始位置
- `_prefill_len: Dict[int, int]` - 每个 sequence 的 prefill 长度

这些变量在 `deallocate()` 时部分清理，但 `prefilled_blocks` 只是 `discard()` 单个 block。

## Hypothesis: Root Cause Chain

```
Request A 完成
    ↓
deallocate() 被调用
    ↓
offload_engine.reset() 被调用
    ↓
GPU buffers 清零 ✅
CPU cache 未清零 ❌  ← 问题点
    ↓
Request B 开始
    ↓
CPU cache 可能包含 Request A 的残留数据
    ↓
错误的 attention 计算
    ↓
错误的输出
```

## 验证策略：状态一致性对比

**核心思路**：对比 fresh-llm 模式和 batch 模式下，同一个 sample 开始时的状态是否一致。

### 需要检查的状态

| 组件 | 状态 | 检查方法 |
|------|------|----------|
| OffloadEngine | `k_cache_cpu`, `v_cache_cpu` | `.sum()` 或 `.abs().max()` |
| OffloadEngine | `k_cache_gpu`, `v_cache_gpu` | `.sum()` 或 `.abs().max()` |
| OffloadEngine | `decode_k/v_buffer` | `.sum()` |
| OffloadEngine | `prefill_k/v_buffer` | `.sum()` |
| HybridManager | `prefilled_blocks` | `len()` |
| HybridManager | `free_logical_ids` | `len()` |
| HybridManager | `free_cpu_blocks` | `len()` |

### 状态检查代码

```python
def dump_state(offload_engine, hybrid_manager, label=""):
    """Dump state for comparison."""
    state = {
        # OffloadEngine GPU state
        "k_gpu_sum": offload_engine.k_cache_gpu.sum().item(),
        "v_gpu_sum": offload_engine.v_cache_gpu.sum().item(),
        # OffloadEngine CPU state
        "k_cpu_sum": offload_engine.k_cache_cpu.sum().item(),
        "v_cpu_sum": offload_engine.v_cache_cpu.sum().item(),
        # Buffers
        "decode_k_sum": offload_engine.decode_k_buffer.sum().item(),
        "decode_v_sum": offload_engine.decode_v_buffer.sum().item(),
        "prefill_k_sum": offload_engine.prefill_k_buffer.sum().item(),
        "prefill_v_sum": offload_engine.prefill_v_buffer.sum().item(),
        # HybridManager
        "prefilled_blocks": len(hybrid_manager.prefilled_blocks),
        "free_logical": len(hybrid_manager.free_logical_ids),
        "free_cpu": len(hybrid_manager.free_cpu_blocks),
    }
    print(f"[STATE {label}] {state}")
    return state

def compare_states(s1, s2):
    """Compare two states, return differences."""
    diffs = {}
    for k in s1:
        if s1[k] != s2[k]:
            diffs[k] = (s1[k], s2[k])
    return diffs
```

### 验证步骤

1. **fresh-llm 模式**：记录 sample N 开始时的状态 (S_fresh)
2. **batch 模式**：记录 sample N 开始时的状态 (S_batch)
3. **对比**：`compare_states(S_fresh, S_batch)`
4. **结论**：差异项即为泄漏源
