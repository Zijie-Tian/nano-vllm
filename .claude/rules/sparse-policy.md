# Sparse Policy 代码规范

## supports_prefill / supports_decode 标志

每个 SparsePolicy 子类必须正确设置这两个标志：

```python
class MyPolicy(SparsePolicy):
    supports_prefill = True   # 是否支持 prefill 阶段
    supports_decode = False   # 是否支持 decode 阶段
```

## 方法实现规范

### 规则：不支持的阶段必须 assert False

如果 policy 不支持某个阶段，对应的 `compute_chunked_*` 方法内部**必须** `assert False`：

```python
class PrefillOnlyPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False

    def compute_chunked_prefill(self, ...):
        # 正常实现 prefill 逻辑
        ...

    def compute_chunked_decode(self, ...):
        # 不支持 decode，必须 assert False
        assert False, "PrefillOnlyPolicy does not support decode phase"
```

```python
class DecodeOnlyPolicy(SparsePolicy):
    supports_prefill = False
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        # 不支持 prefill，必须 assert False
        assert False, "DecodeOnlyPolicy does not support prefill phase"

    def compute_chunked_decode(self, ...):
        # 正常实现 decode 逻辑
        ...
```

### 规则：FullPolicy 必须同时支持两个阶段

`FullAttentionPolicy` 作为默认策略，必须同时支持 prefill 和 decode：

```python
class FullAttentionPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        # 完整实现

    def compute_chunked_decode(self, ...):
        # 完整实现
```

## 调用方检查

`attention.py` 中应在调用前检查 policy 是否支持当前阶段：

```python
# Prefill 路径
if not sparse_policy.supports_prefill:
    raise RuntimeError(f"{sparse_policy} does not support prefill")

# Decode 路径
if not sparse_policy.supports_decode:
    raise RuntimeError(f"{sparse_policy} does not support decode")
```

这样提供双重保护：
1. 调用方检查 → 提供清晰的错误信息
2. 方法内 assert → 防止绕过检查的调用

## CPU-GPU 通信规范

### 规则：所有通信必须通过 OffloadEngine

在 SparsePolicy 的 `compute_chunked_*` 方法中，所有 CPU-GPU 数据传输**必须**通过 `OffloadEngine` 进行，**禁止**直接使用 `torch.Tensor.copy_()` 或 `.to(device)`：

```python
# ✅ 正确：使用 OffloadEngine 的 ring buffer 方法
offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)
k, v = offload_engine.get_kv_for_slot(slot)

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
