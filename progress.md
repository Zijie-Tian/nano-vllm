# Progress Log: Fix Torch Distributed Port Conflict

## Status: COMPLETED & CLEANED UP

## Session: 2026-01-12

### Task Overview
修复在同一 Python 进程中顺序创建多个 LLM 实例时的 EADDRINUSE 端口冲突问题，以及支持多卡环境下同时启动多个独立进程。

---

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | ModelRunner 动态端口分配 | COMPLETED |
| Phase 2 | LLMEngine close() 和 context manager | COMPLETED |
| Phase 3 | 测试验证（GPU 4,5） | COMPLETED |
| Phase 4 | 更新文档 | COMPLETED |

---

### Implementation Summary

#### Phase 1: Dynamic Port Allocation
**File**: `nanovllm/engine/model_runner.py`
- Added `_find_free_port()` function using socket binding
- Modified port selection logic: use env var if set, otherwise auto-assign
- Added logging for auto-assigned ports

#### Phase 2: Resource Cleanup Enhancement
**File**: `nanovllm/engine/llm_engine.py`
- Added `_closed` flag for idempotent cleanup
- Added `close()` method for explicit resource release
- Added `__del__()` for GC fallback
- Added `__enter__()` and `__exit__()` for context manager support
- Modified atexit registration to use `_atexit_handler`

#### Phase 3: Testing (GPU 4,5)
**File**: `tests/test_port_conflict.py`
- Created comprehensive test script

**Test Results**:
| Test | Status | Notes |
|------|--------|-------|
| Sequential creation (3 instances) | PASSED | Ports: 50405, 47835, 53011 |
| Context manager | PASSED | Auto-cleanup works |
| Parallel processes (GPU 4,5) | PASSED | Ports: 34631, 56097 |

#### Phase 4: Documentation
**File**: `docs/torch_distributed_port_issue.md`
- Updated status to RESOLVED
- Documented solution details
- Added usage examples

---

### Files Modified

| File | Action | Description |
|------|--------|-------------|
| `nanovllm/engine/model_runner.py` | Modified | Added `_find_free_port()`, dynamic port logic |
| `nanovllm/engine/llm_engine.py` | Modified | Added `close()`, `__del__`, context manager |
| `tests/test_port_conflict.py` | Created | Test script for port conflict fix |
| `docs/torch_distributed_port_issue.md` | Deleted | Issue resolved, doc removed |
| `CLAUDE.md` | Modified | Removed port conflict warnings, updated doc index |

---

### Key Features After Fix

1. **Multi-GPU Parallel Testing**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python test1.py &
   CUDA_VISIBLE_DEVICES=1 python test2.py &
   # Both run with different auto-assigned ports
   ```

2. **Sequential LLM Creation**
   ```python
   for i in range(3):
       with LLM(model_path) as llm:
           outputs = llm.generate(prompts, params)
       # Automatically cleaned up
   ```

3. **Backward Compatible**
   - `NANOVLLM_DIST_PORT` env var still works
   - `llm.exit()` still works (alias for `close()`)
