# Findings: Torch Distributed Port Conflict

## Problem Analysis

### Issue Summary
创建多个 LLM 实例时出现端口冲突 (EADDRINUSE)，导致第二个实例无法启动。

### Root Cause Deep Dive

#### 1. 资源绑定位置
```python
# nanovllm/engine/model_runner.py:30-32
import os
port = os.environ.get("NANOVLLM_DIST_PORT", "2333")
dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
```

- 默认端口 **2333**，可通过 `NANOVLLM_DIST_PORT` 环境变量配置
- `init_process_group()` 绑定 TCP 端口用于进程间通信
- 端口绑定持续到 `destroy_process_group()` 被调用

#### 2. 清理机制缺陷
```python
# nanovllm/engine/llm_engine.py:37
atexit.register(self.exit)

# nanovllm/engine/llm_engine.py:39-43
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()

# nanovllm/engine/model_runner.py:66-78
def exit(self):
    # ... cleanup code ...
    dist.destroy_process_group()
```

**关键问题**: `atexit` 只在 **Python 解释器退出** 时触发，而非对象被删除时！

#### 3. 问题时间线
```
1. 创建 LLM #1
   ├── init_process_group() 绑定端口 2333 ✓
   └── atexit.register(self.exit) 注册

2. LLM #1 超出作用域或被 del
   ├── Python GC 回收对象内存
   ├── atexit handler 未触发（进程未退出）
   ├── Worker 进程仍在运行
   └── 端口 2333 仍被占用 ❌

3. 创建 LLM #2
   ├── init_process_group() 尝试绑定端口 2333
   └── EADDRINUSE 错误 ❌

4. 程序退出（此时 atexit 才运行）
   └── 为时已晚 - 已经崩溃
```

---

## Solution Analysis

### 方案对比

| 方案 | 可靠性 | 向后兼容 | 实现复杂度 | 推荐度 |
|------|--------|----------|------------|--------|
| `close()` 方法 | 最高 | 是 | 低 | ★★★★★ |
| `__del__` 方法 | 中等 | 是 | 低 | ★★★☆☆ |
| 端口检测重试 | 中等 | 是 | 低 | ★★★☆☆ |
| Context Manager | 最高 | 需要代码修改 | 低 | ★★★★☆ |
| 动态端口 | 低 | 是 | 低 | ★★☆☆☆ |

### 为什么选择三层防护

1. **Layer 1: close()** - 用户显式控制，最可靠
2. **Layer 2: __del__** - 自动清理，覆盖大部分场景
3. **Layer 3: 端口检测** - 最后防线，提供清晰错误信息

### `__del__` 的限制

Python 的 `__del__` 不保证被调用：
- 循环引用时可能不触发
- 解释器关闭时可能无法访问依赖模块
- 不应依赖于 `__del__` 进行关键资源清理

但作为**额外防护层**是有价值的，因为：
- 大多数情况下会被调用
- 比没有好
- 不影响其他清理机制

---

## Code Structure Analysis

### LLMEngine 生命周期
```
__init__()
├── 创建 worker 进程 (self.ps)
├── 创建 ModelRunner (self.model_runner)
├── 注册 atexit handler
└── 设置 scheduler, tokenizer

close() [新增]
├── 检查 _closed 标志（幂等）
├── 注销 atexit handler
├── 调用 model_runner.exit()
├── join worker 进程
└── 设置 _closed = True

__del__() [新增]
└── 调用 close()（忽略异常）

__enter__/__exit__() [新增]
└── Context manager 支持
```

### ModelRunner 资源
```
__init__()
├── torch.distributed 初始化（绑定端口）
├── 模型加载
├── KV cache 分配
├── CUDA graph 捕获（可选）
└── SharedMemory 创建（多GPU）

exit()
├── SharedMemory 清理
├── CUDA graph 清理
└── dist.destroy_process_group()
```

---

## Risk Assessment

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| `__del__` 不被调用 | 中 - 端口泄漏 | Layer 3 端口检测提供清晰错误 |
| close() 重复调用 | 低 | `_closed` 标志保证幂等 |
| atexit 双重调用 | 低 | 注销机制防止 |
| 子进程残留 | 高 | join() 确保子进程退出 |
| CUDA 资源泄漏 | 中 | ModelRunner.exit() 清理 |

---

## Implementation Notes

### atexit.unregister 兼容性
- Python 3.7+ 支持
- 需要传入同一个函数对象
- 使用 `self._atexit_handler` 而非 `self.exit` 以便正确注销

### 端口检测方法
```python
def _check_port_available(port: int, host: str = "localhost") -> bool:
    """使用 socket connect_ex 检测端口是否被占用."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # 0 = connected = port in use
    except Exception:
        return True  # 假设可用
```

**注意**: 这种检测存在 TOCTOU (Time-of-check to time-of-use) 竞争条件，但对于我们的用例足够了。
