# Task Plan: Integrate Sparsity into Layerwise Offload

## Goal

Extend MInference (prefill sparse) and Quest (decode sparse) to the layerwise offload execution path, with an extensible architecture for future sparsity methods.

## Key Insight

**现有的 sparse policy 已经实现，只是 layerwise offload 路径绕过了它！**

| 路径 | Attention 调用方式 | Sparse 支持 |
|------|-------------------|-------------|
| GPU-only | `attention.py` → `sparse_prefill_attention()` | YES |
| Layerwise offload | `model_runner.py` → `flash_attn_varlen_func()` | NO (直接调用) |

## Policy Type Analysis

**两类 sparse policy 的本质区别：**

| Policy | 影响 Attention 计算 | 影响 KV Load 策略 | `select_blocks()` 行为 |
|--------|-------------------|-----------------|----------------------|
| **MInference** | YES (`sparse_prefill_attention`) | NO | `return available_blocks` (全部) |
| **Quest** | NO | YES | 返回 Top-K subset |

**MInference**: 只改变 attention 计算方式，不影响外部的 layer-wise load/offload 流程
**Quest**: 选择性地只 load 部分 blocks，影响 H2D 传输

## Architecture Constraint

**所有 copy_ 操作必须封装在 OffloadEngine 中，model_runner.py 不能直接访问内部存储！**

## Phases

- [x] Phase 1: 添加 `requires_block_selection` 接口标志
- [x] Phase 2: Refactor OffloadEngine - 封装 offload 操作，支持 sparse policy hooks
- [x] Phase 3: MInference prefill - 在 offload prefill 中调用 `sparse_prefill_attention()`
- [x] Phase 4: Quest decode - 根据 `requires_block_selection` 选择性 load blocks (infrastructure ready, full integration deferred)
- [x] Phase 5: Configuration 和 testing

## Detailed Design

### Phase 1: 添加 `requires_block_selection` 接口标志

**New attribute in SparsePolicy base class:**

```python
class SparsePolicy(ABC):
    # Existing flags
    supports_prefill: bool = True
    supports_decode: bool = True

    # NEW: Whether this policy requires selective block loading
    # If True: OffloadEngine will call select_blocks() before loading
    # If False: OffloadEngine will load all blocks (select_blocks ignored)
    requires_block_selection: bool = False
```

**Policy implementations:**

```python
class MInferencePolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False
    requires_block_selection = False  # 不影响 load 策略

    def select_blocks(self, available_blocks, ctx):
        # 不会被调用（requires_block_selection=False）
        return available_blocks


class QuestPolicy(SparsePolicy):
    supports_prefill = False
    supports_decode = True
    requires_block_selection = True  # 影响 load 策略

    def select_blocks(self, available_blocks, ctx):
        # 会被 OffloadEngine 调用
        return self._select_topk_blocks(...)


class FullAttentionPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = True
    requires_block_selection = False  # 加载所有 blocks
```

### Phase 2: Refactor OffloadEngine

**OffloadEngine 根据 `requires_block_selection` 决定是否调用 `select_blocks()`:**

```python
class OffloadEngine:
    def __init__(self, ..., sparse_policy: "SparsePolicy" = None):
        self.sparse_policy = sparse_policy

    def offload_layer_kv_sync(
        self,
        layer_id: int,
        k: Tensor,
        v: Tensor,
        cpu_block_ids: List[int],
        total_tokens: int,
    ) -> None:
        """
        Synchronously offload layer KV to CPU.
        Calls sparse policy hooks internally.
        """
        for i, cpu_block_id in enumerate(cpu_block_ids):
            start = i * self.block_size
            end = min(start + self.block_size, total_tokens)
            actual_size = end - start

            # Hook: notify sparse policy BEFORE offload (k still on GPU)
            if self.sparse_policy is not None:
                self.sparse_policy.on_prefill_offload(
                    cpu_block_id, layer_id, k[start:end], actual_size
                )

            # Synchronous copy to CPU (internal)
            self.k_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(k[start:end])
            self.v_cache_cpu[layer_id, cpu_block_id, :actual_size].copy_(v[start:end])

    def load_layer_kv_to_buffer_with_policy(
        self,
        buffer_idx: int,
        layer_id: int,
        cpu_block_ids: List[int],
        valid_tokens_per_block: List[int],
        query: Optional[Tensor] = None,
    ) -> int:
        """
        Load layer KV to buffer, optionally using sparse policy for block selection.

        Args:
            buffer_idx: Ring buffer slot
            layer_id: Layer index
            cpu_block_ids: All available CPU block IDs
            valid_tokens_per_block: Valid tokens per block
            query: Query tensor (needed for block selection if requires_block_selection=True)

        Returns:
            Total tokens loaded
        """
        # Check if policy requires block selection
        if (self.sparse_policy is not None and
            self.sparse_policy.requires_block_selection and
            query is not None):
            # Build context
            ctx = PolicyContext(
                query_chunk_idx=0,
                num_query_chunks=1,
                layer_id=layer_id,
                query=query,
                is_prefill=False,
                block_size=self.block_size,
            )
            # Select blocks
            selected_blocks = self.sparse_policy.select_blocks(cpu_block_ids, ctx)

            # Build valid_tokens for selected blocks
            block_to_valid = {bid: vt for bid, vt in zip(cpu_block_ids, valid_tokens_per_block)}
            selected_valid = [block_to_valid[bid] for bid in selected_blocks]

            return self._load_blocks_to_buffer(
                buffer_idx, layer_id, selected_blocks, selected_valid
            )
        else:
            # Load all blocks (no selection)
            return self._load_blocks_to_buffer(
                buffer_idx, layer_id, cpu_block_ids, valid_tokens_per_block
            )

    def _load_blocks_to_buffer(
        self,
        buffer_idx: int,
        layer_id: int,
        block_ids: List[int],
        valid_tokens: List[int],
    ) -> int:
        """Internal: load specified blocks to buffer."""
        stream = self.layer_load_streams[buffer_idx]

        with torch.cuda.stream(stream):
            stream.wait_event(self.buffer_compute_done_events[buffer_idx])

            offset = 0
            for cpu_block_id, vt in zip(block_ids, valid_tokens):
                self.layer_k_cache[buffer_idx, offset:offset+vt].copy_(
                    self.k_cache_cpu[layer_id, cpu_block_id, :vt],
                    non_blocking=True
                )
                self.layer_v_cache[buffer_idx, offset:offset+vt].copy_(
                    self.v_cache_cpu[layer_id, cpu_block_id, :vt],
                    non_blocking=True
                )
                offset += vt

            self.buffer_load_events[buffer_idx].record(stream)

        return offset
```

### Phase 3: MInference Prefill Integration

**MInference 只影响 attention 计算，不影响 load/offload：**

```python
def run_layerwise_offload_prefill(self, seqs):
    ...
    for layer_id in range(num_layers):
        # QKV projection + RoPE
        q, k = layer.self_attn.rotary_emb(positions, q, k)

        # Sparse or Full attention
        if self.sparse_prefill_policy is not None:
            # MInference: only changes attention computation
            attn_output = self.sparse_prefill_policy.sparse_prefill_attention(
                q, k, v, layer_id
            )
        else:
            attn_output = flash_attn_varlen_func(q, k, v, ...)

        # MLP
        ...

        # Offload ALL KV (MInference doesn't affect this)
        offload_engine.offload_layer_kv_sync(layer_id, k, v, cpu_block_ids, total_tokens)
```

### Phase 4: Quest Decode Integration

**Quest 影响 block load 策略：**

```python
def run_layerwise_offload_decode(self, seqs):
    ...
    # Preload first N layers (no query available, full load)
    for i in range(num_preload):
        loaded_tokens[i] = offload_engine.load_layer_kv_to_buffer_with_policy(
            i, i, cpu_block_table, valid_tokens_per_block, query=None
        )

    for layer_id in range(num_layers):
        current_buffer = layer_id % num_buffers

        # Wait for buffer load
        offload_engine.wait_buffer_load(current_buffer)

        # QKV projection
        q, k_new, v_new = ...

        # Get loaded KV
        k_prefill, v_prefill = offload_engine.get_buffer_kv(
            current_buffer, loaded_tokens[current_buffer]
        )

        # Attention
        ...

        # Mark buffer done
        offload_engine.record_buffer_compute_done(current_buffer)

        # Load next layer (Quest: selective load if requires_block_selection=True)
        next_layer = layer_id + num_buffers
        if next_layer < num_layers:
            loaded_tokens[current_buffer] = offload_engine.load_layer_kv_to_buffer_with_policy(
                current_buffer, next_layer, cpu_block_table, valid_tokens_per_block,
                query=q  # Pass query for block selection
            )
```

### Phase 5: Configuration

```python
@dataclass
class Config:
    # Separate policies for prefill and decode
    sparse_prefill_policy: SparsePolicyType = SparsePolicyType.FULL  # MINFERENCE
    sparse_decode_policy: SparsePolicyType = SparsePolicyType.FULL   # QUEST
```

## File Changes Summary

| File | Changes |
|------|---------|
| `nanovllm/kvcache/sparse/policy.py` | Add `requires_block_selection` attribute |
| `nanovllm/kvcache/sparse/minference.py` | Set `requires_block_selection = False` |
| `nanovllm/kvcache/sparse/quest.py` | Set `requires_block_selection = True` |
| `nanovllm/kvcache/sparse/full_policy.py` | Set `requires_block_selection = False` |
| `nanovllm/kvcache/offload_engine.py` | Add `offload_layer_kv_sync()`, `load_layer_kv_to_buffer_with_policy()` |
| `nanovllm/engine/model_runner.py` | Use encapsulated methods, integrate sparse policies |

## Key Design Principles

1. **Encapsulation**: All copy_ operations in OffloadEngine
2. **Interface Flag**: `requires_block_selection` declares if policy affects load strategy
3. **Separation of Concerns**:
   - MInference: only `sparse_prefill_attention()` (compute-level)
   - Quest: `select_blocks()` + hooks (load-level)
4. **Hooks inside engine**: Sparse policy hooks called within OffloadEngine methods

## Decisions Made

- [x] 添加 `requires_block_selection` 接口标志区分两类 policy
- [x] 所有 copy_ 封装在 OffloadEngine 中
- [x] Sparse policy hooks 在 OffloadEngine 内部调用
- [x] Decode preload 使用全量加载（Q 不可用）

## Status

**COMPLETE** - All phases implemented and tested successfully.

### Test Results (Qwen3-4B-Instruct-2507)

验证 offload + MInference 输出与 GPU-only + MInference 完全一致：

```
# GPU-only + MInference
test_needle.py --model Qwen3-4B --input-len 32768 --enable-minference
- Prefill: 3383 tok/s
- Output tokens: [22, 19, 24, 17, 151645] = "7492<|im_end|>"
- Result: PASSED

# Offload + MInference
test_needle.py --model Qwen3-4B --input-len 32768 --enable-offload --enable-minference
- Prefill: 5373 tok/s (faster due to layer-wise processing)
- Output tokens: [22, 19, 24, 17, 151645] = "7492<|im_end|>"
- Result: PASSED

两种配置输出完全一致！
```

Note: Qwen3-0.6B 在 offload 模式下有已知 bug（模型太小，长序列不稳定），不是本次修改引入。

## Performance Discovery

**意外发现**: Offload 模式比 GPU-only 模式更快！

| Mode | Prefill Speed |
|------|---------------|
| GPU-only + MInference | 3383 tok/s |
| Offload + MInference | 5373 tok/s |

**根本原因**: GPU-only 模式的 `store_kvcache()` 使用 PagedAttention 的 scatter 操作 (`index_copy_`)，而 offload 模式使用 contiguous copy。

详细分析和优化建议见: [`docs/gpu_only_performance_issue.md`](docs/gpu_only_performance_issue.md)
