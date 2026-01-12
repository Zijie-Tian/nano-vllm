# Torch Distributed Port Conflict Issue

## Problem Summary

When attempting to create multiple `LLM` instances sequentially in the same Python process (e.g., for grouped testing), the second and subsequent instances fail with:

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address.
port: 2333, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use
```

## Root Cause Analysis

### 1. Distributed Process Group Initialization

In `nanovllm/engine/model_runner.py:30-32`:

```python
import os
port = os.environ.get("NANOVLLM_DIST_PORT", "2333")
dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
```

- Default port is **2333** (configurable via `NANOVLLM_DIST_PORT` env var)
- `init_process_group()` binds a TCP socket to this port
- This binding persists until `destroy_process_group()` is called

### 2. Cleanup Mechanism

In `nanovllm/engine/llm_engine.py:37`:

```python
atexit.register(self.exit)
```

In `nanovllm/engine/llm_engine.py:39-43`:

```python
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()
```

In `nanovllm/engine/model_runner.py:66-78`:

```python
def exit(self):
    # ... cleanup code ...
    dist.destroy_process_group()
```

### 3. The Problem

**`atexit` only triggers when the Python interpreter exits, NOT when the object is deleted or goes out of scope.**

Timeline of the bug:

```
1. Create LLM instance #1
   ├── init_process_group() binds port 2333 ✓
   └── atexit.register(self.exit) registered

2. LLM #1 goes out of scope (garbage collected)
   ├── Python's GC deletes the object
   ├── BUT atexit handler NOT triggered yet
   └── Port 2333 still bound! ❌

3. Create LLM instance #2
   ├── init_process_group() tries to bind port 2333
   └── EADDRINUSE error! ❌

4. Program exits (only now atexit runs)
   └── Too late - already crashed
```

## Impact

This issue affects:

1. **Grouped testing mode** (`test_ruler_niah.py --group-size N`)
   - Each group needs a fresh LLM instance
   - Second group fails with port conflict

2. **Multiple LLM instances in same process**
   - Any code that creates LLM, deletes it, then creates another

3. **Interactive/notebook usage**
   - Re-running cells that create LLM instances

## Proposed Solutions

### Solution A: Add `__del__` Method (Quick Fix)

Add destructor to `LLMEngine` that calls cleanup:

```python
# In nanovllm/engine/llm_engine.py

def __del__(self):
    try:
        self.exit()
    except Exception:
        pass  # Ignore errors during cleanup
```

**Pros**: Simple, backwards compatible
**Cons**: `__del__` is not guaranteed to be called (circular references, etc.)

### Solution B: Context Manager Pattern (Recommended)

Make `LLMEngine` a context manager:

```python
# In nanovllm/engine/llm_engine.py

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.exit()
    return False
```

Usage:
```python
with LLM(model_path) as llm:
    outputs = llm.generate(prompts, params)
# Cleanup happens automatically here
```

**Pros**: Explicit, guaranteed cleanup, Pythonic
**Cons**: Requires usage pattern change

### Solution C: Check and Cleanup Before Init (Defensive)

In `ModelRunner.__init__`, check if process group exists:

```python
# In nanovllm/engine/model_runner.py

if dist.is_initialized():
    dist.destroy_process_group()
dist.init_process_group("nccl", f"tcp://localhost:{port}", ...)
```

**Pros**: Self-healing, no usage pattern change
**Cons**: May mask other issues, global state manipulation

### Solution D: Subprocess Isolation (For Testing)

For grouped testing specifically, run each group in a subprocess:

```python
import subprocess
for group in groups:
    subprocess.run([sys.executable, "test_ruler_niah.py",
                    "--sample-indices", f"{start}-{end}"])
```

**Pros**: Complete isolation, no code changes to nanovllm
**Cons**: More overhead, only solves testing use case

### Solution E: Dynamic Port Allocation

Instead of fixed port 2333, use dynamic port:

```python
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = os.environ.get("NANOVLLM_DIST_PORT") or find_free_port()
```

**Pros**: Avoids conflicts entirely
**Cons**: More complex, may have side effects

## Recommended Implementation

**Combine Solutions A + B + C** for maximum robustness:

1. Add `__del__` for best-effort cleanup
2. Add context manager for explicit cleanup
3. Add `is_initialized()` check as defensive measure

```python
# nanovllm/engine/llm_engine.py

class LLMEngine:
    def __init__(self, model, **kwargs):
        # ... existing code ...
        atexit.register(self.exit)
        self._exited = False

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def __del__(self):
        try:
            self.exit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exit()
        return False


# nanovllm/engine/model_runner.py

class ModelRunner:
    def __init__(self, config: Config, rank: int, event):
        # ... existing code before init_process_group ...

        import os
        port = os.environ.get("NANOVLLM_DIST_PORT", "2333")

        # Defensive cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

        dist.init_process_group("nccl", f"tcp://localhost:{port}",
                                world_size=self.world_size, rank=rank)
        # ... rest of init ...
```

## Workaround for Current Code

Until the fix is implemented, use one of these workarounds:

### Workaround 1: Manual Cleanup

```python
import torch.distributed as dist

llm = LLM(model_path)
outputs = llm.generate(...)
llm.model_runner.call("exit")  # Manual cleanup
del llm

# Now can create new LLM
llm2 = LLM(model_path)
```

### Workaround 2: Subprocess Testing

```bash
# Run each test group as separate process
for i in $(seq 0 5 95); do
    python test_ruler_niah.py --sample-indices $i-$((i+4)) --enable-offload
done
```

### Workaround 3: Environment Variable Port

```bash
# Use different port for each run
NANOVLLM_DIST_PORT=2334 python test.py
NANOVLLM_DIST_PORT=2335 python test.py
```

## Related Files

| File | Relevant Code |
|------|---------------|
| `nanovllm/engine/model_runner.py:30-32` | `init_process_group()` call |
| `nanovllm/engine/model_runner.py:66-78` | `exit()` and `destroy_process_group()` |
| `nanovllm/engine/llm_engine.py:37` | `atexit.register()` |
| `nanovllm/engine/llm_engine.py:39-43` | `exit()` method |

## Testing the Fix

After implementing the fix, verify with:

```python
# test_multiple_llm.py
from nanovllm import LLM, SamplingParams

for i in range(3):
    print(f"Creating LLM instance {i+1}")
    llm = LLM("path/to/model", enable_cpu_offload=True)
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"Instance {i+1} output: {outputs[0]['text']}")
    del llm
    print(f"Instance {i+1} deleted\n")

print("All instances created and deleted successfully!")
```

Expected: No port conflict errors, all 3 instances work.

## Priority

**High** - This blocks grouped testing and any multi-LLM-instance workflows.
