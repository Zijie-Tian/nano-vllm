# Sparse Policy 代码规范

## 基类要求 (MANDATORY)

每个 `SparsePolicy` 子类 **必须** 遵守以下要求：

### 1. 声明 supports_prefill / supports_decode 标志

```python
class MyPolicy(SparsePolicy):
    supports_prefill = True   # 是否支持 prefill 阶段
    supports_decode = True    # 是否支持 decode 阶段
```

### 2. 实现三个抽象方法

| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `select_blocks()` | ✅ | 选择要加载的 blocks |
| `compute_chunked_prefill()` | ✅ | Prefill attention 计算 |
| `compute_chunked_decode()` | ✅ | Decode attention 计算 |

### 3. 不支持的阶段必须 assert False

如果 `supports_prefill = False`，则 `compute_chunked_prefill()` 内部 **必须** `assert False`：

```python
class DecodeOnlyPolicy(SparsePolicy):
    supports_prefill = False
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        assert False, "DecodeOnlyPolicy does not support prefill phase"

    def compute_chunked_decode(self, ...):
        # 正常实现
        ...
```

同理，如果 `supports_decode = False`：

```python
class PrefillOnlyPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False

    def compute_chunked_prefill(self, ...):
        # 正常实现
        ...

    def compute_chunked_decode(self, ...):
        assert False, "PrefillOnlyPolicy does not support decode phase"
```

### 4. FullAttentionPolicy 必须同时支持两个阶段

```python
class FullAttentionPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        # 完整实现

    def compute_chunked_decode(self, ...):
        # 完整实现
```

---

## CPU-GPU 通信规范

### 规则：所有通信必须通过 OffloadEngine

在 `compute_chunked_*` 方法中，**禁止** 直接使用 `torch.Tensor.copy_()` 或 `.to(device)`：

```python
# ✅ 正确：使用 OffloadEngine 的 ring buffer 方法
offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)
k, v = offload_engine.get_kv_for_slot(slot)
offload_engine.record_slot_compute_done(slot)

# ✅ 正确：使用 prefill buffer
k, v = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)

# ✅ 正确：使用 decode buffer
decode_k = offload_engine.decode_k_buffer[layer_id, start:end]
decode_v = offload_engine.decode_v_buffer[layer_id, start:end]

# ❌ 错误：直接使用 torch 通信
gpu_tensor.copy_(cpu_tensor)
gpu_tensor = cpu_tensor.to("cuda")
gpu_tensor = cpu_tensor.cuda()
```

### 原因

1. **流同步**：OffloadEngine 内部管理 CUDA streams，确保正确的同步
2. **Pipeline 优化**：OffloadEngine 实现了 ring buffer pipeline
3. **资源管理**：OffloadEngine 管理 GPU buffer slots，避免内存碎片
4. **一致性**：统一的接口便于调试和维护

---

## 方法签名要求

### select_blocks()

```python
def select_blocks(
    self,
    available_blocks: List[int],      # 可用的 CPU block IDs
    offload_engine: "OffloadEngine",  # 用于加载数据
    ctx: PolicyContext,               # 上下文信息
) -> List[int]:                       # 返回要加载的 block IDs
```

### compute_chunked_prefill()

```python
def compute_chunked_prefill(
    self,
    q: torch.Tensor,                  # [seq_len, num_heads, head_dim]
    k: torch.Tensor,                  # [seq_len, num_kv_heads, head_dim] (unused)
    v: torch.Tensor,                  # [seq_len, num_kv_heads, head_dim] (unused)
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    current_chunk_idx: int,
    seq: "Sequence",
    num_tokens: int,
) -> torch.Tensor:                    # [seq_len, num_heads, head_dim]
```

### compute_chunked_decode()

```python
def compute_chunked_decode(
    self,
    q: torch.Tensor,                  # [batch_size, num_heads, head_dim]
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    seq: "Sequence",
) -> torch.Tensor:                    # [batch_size, 1, num_heads, head_dim]
```

---

## 可选钩子方法

| 方法 | 调用时机 | 用途 |
|------|---------|------|
| `initialize()` | KV cache 分配后 | 初始化 metadata 结构 |
| `on_prefill_offload()` | GPU→CPU 复制前（prefill） | 收集 block metadata |
| `on_decode_offload()` | GPU→CPU 复制前（decode） | 更新 block metadata |
| `reset()` | 新 sequence 开始时 | 重置 policy 状态 |

---

## 详细实现指南

参考文档：[`docs/sparse_policy_implementation_guide.md`](../docs/sparse_policy_implementation_guide.md)
