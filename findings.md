# Findings: nanovllm 多请求状态污染分析

## 重要说明

**nanovllm offload 模式不支持 batch**，只能单个 request 顺序执行。问题出在**请求切换**（前一个 request 完成后，开始下一个 request）时状态清理不完整。

---

## 1. 代码架构发现

### 1.1 请求生命周期 (顺序执行)

**关键**: offload 模式下，每次只处理**一个 request**，不是 batch。

```
LLMEngine.generate() [llm_engine.py:114-151]
├── Observer.complete_reset()  # 重置性能统计
├── for prompt in prompts:
│   └── add_request(prompt, sp)  # 添加到 scheduler 队列
├── while not is_finished():
│   ├── scheduler.schedule()  # 获取下一个序列 (offload 模式: 1个)
│   ├── model_runner.call("run", seqs, is_prefill)  # 执行单个请求
│   └── scheduler.postprocess(seqs, token_ids)
│       └── if seq.is_finished:
│           └── kvcache_manager.deallocate(seq)  # 释放资源 ← 问题点
│           └── [开始处理下一个请求]  # ← 状态切换
└── return outputs
```

**请求切换流程**:
```
Request A (prefill) → Request A (decode × N) → Request A 完成
    ↓
deallocate(A)  ← 状态清理不完整!
    ↓
Request B (prefill) → Request B 读取到 A 的残留状态 → 错误输出
```

### 1.2 OffloadEngine 状态清单

**位置**: `nanovllm/kvcache/offload_engine.py:40-145`

| 成员变量 | 类型 | Shape | 生命周期 |
|----------|------|-------|----------|
| `layer_k_cache` | GPU Tensor | [num_buffers, max_seq_len, kv_heads, head_dim] | 整个引擎 |
| `layer_v_cache` | GPU Tensor | [num_buffers, max_seq_len, kv_heads, head_dim] | 整个引擎 |
| `decode_k_buffer` | GPU Tensor | [num_layers, block_size, kv_heads, head_dim] | 整个引擎 |
| `decode_v_buffer` | GPU Tensor | [num_layers, block_size, kv_heads, head_dim] | 整个引擎 |
| `k_cache_cpu` | CPU Tensor (pinned) | [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim] | 整个引擎 |
| `v_cache_cpu` | CPU Tensor (pinned) | [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim] | 整个引擎 |
| `compute_stream` | CUDA Stream | - | 整个引擎 |
| `prefill_offload_streams` | List[CUDA Stream] | num_layers | 整个引擎 |
| `prefill_offload_events` | List[CUDA Event] | num_layers | 整个引擎 |
| `layer_load_streams` | List[CUDA Stream] | num_buffers | 整个引擎 |
| `buffer_load_events` | List[CUDA Event] | num_buffers | 整个引擎 |
| `buffer_compute_done_events` | List[CUDA Event] | num_buffers | 整个引擎 |

**关键发现**:
- **没有 reset() 方法**
- **没有任何清理逻辑**
- 所有 tensor 在初始化时 `torch.zeros()` 后永不清零

### 1.3 HybridKVCacheManager 状态清单

**位置**: `nanovllm/kvcache/hybrid_manager.py`

| 成员变量 | 作用 | 清理方式 |
|----------|------|----------|
| `logical_blocks` | 逻辑块列表 | `block.reset()` in deallocate |
| `free_logical_ids` | 空闲逻辑块队列 | deallocate 归还 |
| `free_cpu_blocks` | 空闲 CPU 块队列 | deallocate 归还 |
| `cpu_block_to_logical` | CPU 块→逻辑块映射 | deallocate 删除 |
| `prefilled_blocks` | 已 prefill 的块集合 | deallocate 中 discard |
| `_decode_start_pos` | 序列→decode起始位置 | `clear_decode_tracking()` |
| `_prefill_len` | 序列→prefill长度 | `clear_decode_tracking()` |

**关键发现**:
- `deallocate()` 没有调用 `clear_decode_tracking()`！
- `_decode_start_pos` 和 `_prefill_len` 使用 `id(seq)` 作为 key
- Python 对象 ID 可能在不同请求间重用

---

## 2. 请求切换机制分析

### 2.1 offload 模式的单 request 限制

代码中明确限制：
```python
# model_runner.py:757, 880
assert len(seqs) == 1, "Layer-wise offload only supports single sequence"
```

### 2.2 请求切换时序

```
时间 →
┌─────────────────────────────────────────────────────────────────┐
│ Request A: [prefill] → [decode] → [decode] → ... → [完成]       │
└─────────────────────────────────────────────────────────────────┘
                                                      ↓
                                          deallocate(seq_A)
                                          - blocks 释放 ✓
                                          - tracking 字典未清理 ✗
                                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ Request B: [prefill] → [decode] → ...                           │
│            ↑                                                     │
│            如果 id(seq_B) == id(seq_A)，读到 A 的残留状态！      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Python 对象 ID 重用

Python 的内存管理会重用已释放对象的内存地址，导致：
```python
seq_A = Sequence(...)  # id(seq_A) = 0x7f1234567890
del seq_A              # 对象被释放，但字典中 key 保留

seq_B = Sequence(...)  # id(seq_B) 可能 = 0x7f1234567890（相同地址）
# _decode_start_pos[id(seq_B)] 返回 seq_A 的旧值！
```

---

## 3. 状态污染机制分析

### 3.1 decode buffer 污染路径

**污染写入** (`run_layerwise_offload_decode:1010-1013`):
```python
# 每次 decode step，将当前 token 的 KV 存入 decode buffer
offload_engine.decode_k_buffer[layer_id, pos_in_block].copy_(ring_k[context_len])
offload_engine.decode_v_buffer[layer_id, pos_in_block].copy_(ring_v[context_len])
```

**污染读取** (`run_layerwise_offload_decode:969-976`):
```python
# 如果有之前的 decode tokens，从 decode buffer 读取
if num_prev_decode_tokens > 0:
    k_decode_prev, v_decode_prev = offload_engine.get_decode_kv(
        layer_id, decode_start_pos, pos_in_block
    )
    ring_k[total_prefill_tokens:total_prefill_tokens + num_prev_decode_tokens].copy_(k_decode_prev)
```

**问题场景**:
1. 请求 A 的 decode 阶段在 `decode_k_buffer[layer, 0:N]` 写入 KV
2. 请求 A 完成，buffer 数据保留
3. 请求 B 开始，如果其 `decode_start_pos` 被错误计算为非零
4. 请求 B 会读取请求 A 的旧数据

### 3.2 decode_start_pos 计算逻辑

**位置**: `hybrid_manager.py:485-505`

```python
def get_decode_start_pos(self, seq: Sequence) -> int:
    seq_id = id(seq)  # Python 对象 ID
    if seq_id not in self._decode_start_pos:
        # 第一次调用 - 计算起始位置
        prefill_len = len(seq) - 1  # 当前长度减去新 token
        self._decode_start_pos[seq_id] = prefill_len % self._block_size
    return self._decode_start_pos[seq_id]
```

**问题**:
- 如果新请求的 `id(seq)` 恰好等于旧请求的 `id(seq)`（Python 内存重用）
- `_decode_start_pos` 中可能存在旧的值
- 会返回错误的 decode 起始位置

### 3.3 clear_decode_tracking 未被调用

**位置**: `hybrid_manager.py:538-549`

```python
def clear_decode_tracking(self, seq: Sequence) -> None:
    seq_id = id(seq)
    self._decode_start_pos.pop(seq_id, None)
    self._prefill_len.pop(seq_id, None)
```

**问题**:
- 这个方法在 `deallocate()` 中**没有被调用**！
- 查看 `deallocate()` (218-244 行)，没有 `clear_decode_tracking()` 调用
- 这导致旧请求的 tracking 数据残留

---

## 3. 失败模式分析

### 3.1 观察到的失败模式

从测试结果:
| Sample | Expected | Output | Status |
|--------|----------|--------|--------|
| 0 | 8930103 | `: 8930103.` | PASS (第一个请求) |
| 1 | 4194548 | `: 419 multiplication of 4548.` | **FAIL** |
| 2 | 8231838 | `:ное 8231838.` | PASS |

Sample 1 的输出 "419 multiplication of 4548" 显示数字被"拆分"了。

**可能原因**:
1. 在某个 decode step，attention 计算使用了错误的 KV
2. 模型"看到"了旧请求的部分 context
3. 导致生成逻辑出错

### 3.2 为什么第一个请求总是成功？

1. 第一个请求时，所有 buffer 都是零初始化
2. `decode_start_pos` 字典为空，正确计算
3. 没有残留数据干扰

### 3.3 为什么后续请求可能成功？

某些请求可能成功因为：
1. `id(seq)` 没有与之前的请求冲突
2. `pos_in_block` 不重叠，没读到旧数据
3. 或者旧数据恰好对结果影响不大

---

## 4. 修复方向

### 4.1 必须修复: deallocate 时清理状态

```python
# hybrid_manager.py: deallocate()
def deallocate(self, seq: Sequence) -> None:
    # ... 现有逻辑 ...

    # 添加: 清理 decode tracking
    self.clear_decode_tracking(seq)

    # 添加: 通知 offload engine 清理
    if self.offload_engine is not None:
        self.offload_engine.on_sequence_finished()
```

### 4.2 必须修复: OffloadEngine 添加清理方法

```python
# offload_engine.py
def on_sequence_finished(self):
    """请求完成时的清理"""
    # 清零 decode buffer
    self.decode_k_buffer.zero_()
    self.decode_v_buffer.zero_()
```

### 4.3 可选: 更激进的清理

```python
def reset_all(self):
    """完全重置状态"""
    self.decode_k_buffer.zero_()
    self.decode_v_buffer.zero_()
    self.layer_k_cache.zero_()
    self.layer_v_cache.zero_()
    # 重置 CUDA events
    for event in self.buffer_compute_done_events:
        event.record()
```

---

## 5. 待验证假设

| 假设 | 验证方法 | 优先级 |
|------|----------|--------|
| decode_buffer 残留导致污染 | 在第二个请求开始时检查 buffer 是否为零 | 高 |
| _decode_start_pos 字典残留 | 打印 deallocate 前后的字典内容 | 高 |
| id(seq) 重用导致错误 | 打印每个请求的 seq id | 中 |
| ring buffer 残留 | 检查每次 decode 前 ring buffer 内容 | 低 |

---

## 6. 参考代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| OffloadEngine 初始化 | offload_engine.py | 40-145 |
| deallocate | hybrid_manager.py | 218-244 |
| clear_decode_tracking | hybrid_manager.py | 538-549 |
| get_decode_start_pos | hybrid_manager.py | 485-505 |
| run_layerwise_offload_decode | model_runner.py | 867-1057 |
| decode buffer 写入 | model_runner.py | 1010-1013 |
| decode buffer 读取 | model_runner.py | 969-976 |
