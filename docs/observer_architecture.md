# Observer Architecture

nanovllm 的 Observer 架构用于统计推理过程中的关键指标，采用类变量（class variable）模式实现全局状态管理。

## 架构概览

```
Observer (基类)
├── InferenceObserver - 推理时间指标 (TTFT, TPOT)
└── MemoryObserver - 内存传输统计 (H2D, D2H, D2D)
```

## 设计原则

### 1. 类变量模式

所有 Observer 使用类变量（而非实例变量）存储状态：

```python
class Observer:
    """Observer 基类"""
    _enabled: bool = True  # 类变量，控制是否启用

class InferenceObserver(Observer):
    ttft: int = 0          # 类变量，全局共享
    tpot: int = 0
    ttft_start: int = 0
    tpot_start: int = 0
```

**优点**：
- 无需实例化，任何地方都可以直接访问
- 避免跨模块传递 observer 实例
- 适合全局统计场景

### 2. 启用/禁用控制

每个 Observer 可独立启用/禁用：

```python
# 启用 MemoryObserver
MemoryObserver._enabled = True

# 禁用后，record_* 方法不会记录
MemoryObserver._enabled = False
```

### 3. 阶段分离

MemoryObserver 支持 prefill/decode 阶段分离统计：

```python
@classmethod
def record_h2d(cls, num_bytes: int, is_prefill: bool = True) -> None:
    if not cls._enabled:
        return
    cls.h2d_bytes += num_bytes
    cls.h2d_count += 1
    if is_prefill:
        cls.prefill_h2d_bytes += num_bytes
    else:
        cls.decode_h2d_bytes += num_bytes
```

## Observer 实现

### InferenceObserver

**位置**: `nanovllm/utils/observer.py`

**统计指标**：
| 指标 | 说明 | 单位 |
|------|------|------|
| `ttft` | Time To First Token | 纳秒 |
| `tpot` | Time Per Output Token | 纳秒 |
| `ttft_start` | TTFT 计时开始点 | 纳秒 |
| `tpot_start` | TPOT 计时开始点 | 纳秒 |

**统计位置**：
| 位置 | 代码 | 说明 |
|------|------|------|
| `scheduler.py:add()` | `InferenceObserver.ttft_start = perf_counter_ns()` | 开始计时 |
| `llm_engine.py:step()` | `InferenceObserver.ttft = ... - ttft_start` | Prefill 完成后计算 TTFT |
| `llm_engine.py:step()` | `InferenceObserver.tpot = ... - tpot_start` | Decode 时计算 TPOT |

### MemoryObserver

**位置**: `nanovllm/utils/memory_observer.py`

**统计指标**：
| 指标 | 说明 |
|------|------|
| `h2d_bytes` / `h2d_count` | Host to Device 传输量/次数 |
| `d2h_bytes` / `d2h_count` | Device to Host 传输量/次数 |
| `d2d_bytes` / `d2d_count` | Device to Device 复制量/次数 |
| `prefill_h2d_bytes` / `prefill_d2h_bytes` | Prefill 阶段 H2D/D2H |
| `decode_h2d_bytes` / `decode_d2h_bytes` | Decode 阶段 H2D/D2H |

**统计位置** (均在 `offload_engine.py`)：

| 方法 | 传输类型 | 说明 |
|------|----------|------|
| `load_to_slot_layer()` | H2D | 从 CPU 加载 block 到 GPU slot |
| `offload_slot_layer_to_cpu()` | D2H | GPU slot 卸载到 CPU |
| `offload_prefill_buffer_async()` | D2H | Prefill buffer 异步卸载 |
| `write_to_prefill_buffer()` | D2D | 写入 prefill buffer |
| `write_to_decode_buffer()` | D2D | 写入 decode buffer |

**重置位置**：
| 位置 | 代码 |
|------|------|
| `llm_engine.py:generate()` | `MemoryObserver.complete_reset()` |
| `llm_engine.py:generate()` | `InferenceObserver.complete_reset()` |

## 使用示例

### 1. 启用并统计

```python
from nanovllm.utils.memory_observer import MemoryObserver

# 启用统计
MemoryObserver._enabled = True

# 运行推理
outputs = llm.generate(prompts, sampling_params)

# 获取结果
print(f"Prefill H2D: {MemoryObserver.prefill_h2d_bytes / 1e9:.2f} GB")
print(f"Decode H2D: {MemoryObserver.decode_h2d_bytes / 1e9:.2f} GB")

# 或使用 print_summary
MemoryObserver.print_summary()
```

### 2. 在 bench_offload.py 中

```python
from nanovllm.utils.memory_observer import MemoryObserver

# 启用
MemoryObserver._enabled = True

# benchmark 结束后打印
def print_memory_stats():
    fmt = MemoryObserver._fmt_bytes
    print(f"[Memory] Prefill H2D: {fmt(MemoryObserver.prefill_h2d_bytes)}")
    print(f"         Decode  H2D: {fmt(MemoryObserver.decode_h2d_bytes)}")
```

### 3. 获取结构化数据

```python
summary = MemoryObserver.get_summary()
# {
#     "total": {"h2d_bytes": ..., "d2h_bytes": ..., "d2d_bytes": ...},
#     "prefill": {"h2d_bytes": ..., "d2h_bytes": ...},
#     "decode": {"h2d_bytes": ..., "d2h_bytes": ...}
# }
```

## 添加新 Observer

1. 继承 `Observer` 基类
2. 定义类变量存储统计数据
3. 实现 `record_*` 方法（需检查 `_enabled`）
4. 实现 `complete_reset()` 方法
5. 在相关代码位置添加 `record_*` 调用
6. 在 `llm_engine.py:generate()` 中添加 reset 调用

```python
from nanovllm.utils.observer import Observer

class MyObserver(Observer):
    _enabled: bool = False
    my_metric: int = 0

    @classmethod
    def record_event(cls, value: int) -> None:
        if not cls._enabled:
            return
        cls.my_metric += value

    @classmethod
    def complete_reset(cls) -> None:
        cls.my_metric = 0
```

## 相关文档

- [`memory_communication_benchmark.md`](memory_communication_benchmark.md) - 通信量测试结果
- [`architecture_guide.md`](architecture_guide.md) - 整体架构指南
