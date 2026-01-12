# Task Plan: Fix Torch Distributed Port Conflict

## Goal
支持多卡环境下同时启动多个独立的 nanovllm 进程进行测试，无需手动管理端口。

## Problem Analysis

### 核心问题
```
当前：所有 nanovllm 实例默认使用端口 2333
     └── 多个独立进程同时运行时会冲突！

CUDA_VISIBLE_DEVICES=0 python test1.py  # 绑定端口 2333 ✓
CUDA_VISIBLE_DEVICES=1 python test2.py  # 尝试绑定 2333 → EADDRINUSE ❌
```

### 根本原因
- 端口是系统级资源，与 GPU 无关
- 即使使用不同 GPU，端口仍会冲突
- 当前硬编码默认端口 `2333`

---

## Solution: Dynamic Port Allocation

### 核心方案
```python
def _find_free_port() -> int:
    """让系统自动分配一个空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# 优先使用环境变量，否则自动分配
port = os.environ.get("NANOVLLM_DIST_PORT")
if port is None:
    port = _find_free_port()
else:
    port = int(port)
```

### 效果
```bash
# 无需手动指定端口，可以同时运行多个测试
CUDA_VISIBLE_DEVICES=0 python test1.py &  # 自动端口 54321
CUDA_VISIBLE_DEVICES=1 python test2.py &  # 自动端口 54322
CUDA_VISIBLE_DEVICES=2 python test3.py &  # 自动端口 54323

# 仍然支持手动指定（向后兼容）
NANOVLLM_DIST_PORT=2333 python test.py
```

---

## Implementation Phases

### Phase 1: ModelRunner 动态端口 [pending]
**File**: `nanovllm/engine/model_runner.py`

```python
import socket

def _find_free_port() -> int:
    """Find a free port for distributed communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # ... existing code ...

        import os
        port = os.environ.get("NANOVLLM_DIST_PORT")
        if port is None:
            port = _find_free_port()
            logger.info(f"Auto-assigned distributed port: {port}")
        else:
            port = int(port)

        dist.init_process_group("nccl", f"tcp://localhost:{port}", ...)
```

### Phase 2: LLMEngine 资源清理增强 [pending]
**File**: `nanovllm/engine/llm_engine.py`

添加 `close()` 方法和 context manager 支持，确保资源正确释放：

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # ... existing code ...
        self._closed = False
        atexit.register(self._atexit_handler)

    def _atexit_handler(self):
        if not self._closed:
            self.close()

    def close(self):
        """Explicitly close the engine and release all resources."""
        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self._atexit_handler)
        except Exception:
            pass
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def exit(self):
        """Alias for close() - backward compatibility."""
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
```

### Phase 3: 测试验证 [pending]
**File**: `tests/test_multiple_processes.py` (新建)

```python
"""Test multiple independent nanovllm processes."""
import subprocess
import sys
import time

def test_parallel_processes():
    """Test running multiple nanovllm processes in parallel."""
    script = '''
import sys
sys.path.insert(0, ".")
from nanovllm import LLM, SamplingParams
import os

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(f"[GPU {gpu}] Starting LLM")
llm = LLM("path/to/model", enable_cpu_offload=True)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
print(f"[GPU {gpu}] Output: {outputs[0]['text'][:50]}")
llm.close()
print(f"[GPU {gpu}] Done")
'''

    # Start 2 processes on different GPUs
    procs = []
    for gpu in [0, 1]:
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        p = subprocess.Popen(
            [sys.executable, "-c", script],
            env={**os.environ, **env}
        )
        procs.append(p)
        time.sleep(1)  # Stagger start slightly

    # Wait for all
    for p in procs:
        assert p.wait() == 0, f"Process failed with code {p.returncode}"

    print("PASSED: test_parallel_processes")

if __name__ == "__main__":
    test_parallel_processes()
```

### Phase 4: 文档更新 [pending]
**File**: `docs/torch_distributed_port_issue.md`

更新文档标记问题已通过动态端口分配解决。

---

## Usage After Fix

### 场景 1: 多进程并行测试（主要场景）
```bash
# 无需任何额外配置，直接运行
CUDA_VISIBLE_DEVICES=0 python test_group1.py &
CUDA_VISIBLE_DEVICES=1 python test_group2.py &
CUDA_VISIBLE_DEVICES=2 python test_group3.py &
wait
```

### 场景 2: 同一进程顺序创建（也支持）
```python
for i in range(3):
    with LLM(model_path) as llm:
        outputs = llm.generate(prompts, params)
    # 自动清理，下一个可以使用新的随机端口
```

### 场景 3: 手动指定端口（向后兼容）
```bash
NANOVLLM_DIST_PORT=2333 python test.py
```

---

## Success Criteria

- [ ] 多个独立进程可以同时运行（不同 GPU）
- [ ] 无需手动指定端口
- [ ] 向后兼容（环境变量仍有效）
- [ ] 同一进程顺序创建也能工作
- [ ] 资源正确清理

---

## Files to Modify

| File | Action | Status |
|------|--------|--------|
| `nanovllm/engine/model_runner.py` | Add `_find_free_port()` | pending |
| `nanovllm/engine/llm_engine.py` | Add `close()`, context manager | pending |
| `tests/test_multiple_processes.py` | Create | pending |
| `docs/torch_distributed_port_issue.md` | Update | pending |
