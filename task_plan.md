# Task Plan: Sparse Policy 架构重构 v3

## Goal

将 chunked prefill 的 attention 计算逻辑完全从 `attention.py` 移到 `SparsePolicy` 内部。attention.py 只负责调用 policy，不包含任何计算逻辑。

## 核心设计原则（强制要求）

1. **Policy 内部完成所有计算**：包括 attention 计算和结果合并
2. **select_blocks 传入 offload_engine**：policy 通过 offload_engine 加载 blocks
3. **强制实现计算函数**：所有 policy 必须实现 `compute_block_attention` 和 `merge_attention_outputs`
4. **chunked_prefill 强制 policy 存在**：没有 policy 则报错
5. **外部默认 FULL policy**：model_runner.py 默认创建 FullPolicy
6. **attention.py 零计算逻辑**：_chunked_prefill_attention 只调用 policy，不直接调用 flashattn 或 merge

## 目标架构

```
model_runner.py:
  默认创建 FullPolicy（如果没有指定 sparse policy）

attention.py (_chunked_prefill_attention):
  检查 sparse_policy 是否存在
    ↓
  调用 sparse_policy.compute_prefill_attention(q, k, v, ...)
    ↓
  返回最终输出（不包含任何计算逻辑）

SparsePolicy.compute_prefill_attention():
  1. select_blocks(blocks, offload_engine, ctx) → 筛选 blocks
  2. 加载 blocks（通过 offload_engine）
  3. 遍历 blocks：
     - 调用 self.compute_block_attention(q, k, v, ...)
     - 调用 self.merge_attention_outputs(...)
  4. 计算当前 chunk attention
  5. 合并最终结果
  6. 返回 final_output
```

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **决策 1** | `compute_block_attention` 是抽象方法，所有 policy 必须实现 |
| **决策 2** | `merge_attention_outputs` 是抽象方法，所有 policy 必须实现 |
| **决策 3** | `compute_prefill_attention` 是抽象方法，定义完整的 prefill 流程 |
| **决策 4** | `select_blocks` 接收 `offload_engine` 参数（为未来准备） |
| **决策 5** | chunked_prefill 检查 policy 是否存在，不存在则抛出错误 |
| **决策 6** | model_runner 默认创建 FullPolicy 作为兜底 |
| **决策 7** | attention.py 的 _chunked_prefill_attention 不包含任何 flashattn 或 merge 调用 |

## Phases

- [ ] Phase 1: 分析当前架构，理解所有计算逻辑的位置
- [ ] Phase 2: 在 SparsePolicy 基类中添加三个抽象方法
- [ ] Phase 3: 修改 FullPolicy，实现三个抽象方法
- [ ] Phase 4: 修改 QuestPolicy，实现三个抽象方法
- [ ] Phase 5: 修改 XAttentionBSAPolicy，实现三个抽象方法
- [ ] Phase 6: 修改 model_runner.py，默认创建 FullPolicy
- [ ] Phase 7: 修改 attention.py，移除所有计算逻辑，只调用 policy
- [ ] Phase 8: 测试验证

## Phase 1: 分析当前架构，理解所有计算逻辑的位置

### 当前 attention.py 中包含的计算逻辑

1. `_ring_buffer_pipeline_load` 方法：
   - 调用 `offload_engine.load_to_slot_layer()`
   - 调用 `offload_engine.wait_slot_layer()`
   - 调用 `offload_engine.get_kv_for_slot()`
   - 调用 `flash_attn_with_lse()` ← **直接调用**
   - 调用 `merge_attention_outputs()` ← **直接调用**

2. `_sync_load_previous_chunks` 方法：
   - 同上，直接调用 flashattn 和 merge

3. `_chunked_prefill_attention` 方法：
   - 调用 `_ring_buffer_pipeline_load` 或 `_sync_load_previous_chunks`
   - 调用 `flash_attn_with_lse()` 计算当前 chunk
   - 调用 `merge_attention_outputs()` 合并结果

### 需要移动的计算逻辑

所有 `flash_attn_with_lse` 和 `merge_attention_outputs` 调用都应该在 SparsePolicy 内部。

## Phase 2: 在 SparsePolicy 基类中添加三个抽象方法

### 2.1 compute_block_attention

```python
@abstractmethod
def compute_block_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    计算单个 block 的 attention。

    Args:
        q: [1, seq_len, num_heads, head_dim] 或 [seq_len, num_heads, head_dim]
        k, v: 同上
        layer_id: 层索引
        softmax_scale: softmax 缩放因子
        causal: 是否应用因果掩码

    Returns:
        (o, lse) - attention 输出和 LSE
    """
    pass
```

### 2.2 merge_attention_outputs

```python
@abstractmethod
def merge_attention_outputs(
    self,
    o_acc: torch.Tensor,
    lse_acc: Optional[torch.Tensor],
    o_new: torch.Tensor,
    lse_new: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    合并两个 attention 输出。

    Args:
        o_acc: 累积的 attention 输出 [1, seq_len, num_heads, head_dim]
        lse_acc: 累积的 LSE
        o_new: 新的 attention 输出
        lse_new: 新的 LSE

    Returns:
        (merged_o, merged_lse)
    """
    pass
```

### 2.3 compute_chunked_attention

```python
@abstractmethod
def compute_chunked_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
    softmax_scale: float,
    offload_engine: OffloadEngine,
    current_chunk_idx: int,
    seq: ChunkedSequence,
    num_tokens: int,
) -> torch.Tensor:
    """
    计算 chunked prefill attention（完整流程）。

    这是 policy 的主入口，定义完整的 prefill 计算流程：
    1. 获取历史 blocks
    2. 筛选 blocks（调用 select_blocks）
    3. 加载和计算历史 blocks
    4. 计算当前 chunk attention
    5. 合并所有结果

    Args:
        q, k, v: 当前 chunk 的 QKV
        layer_id: 层索引
        softmax_scale: softmax 缩放因子
        offload_engine: offload engine
        current_chunk_idx: 当前 chunk 索引
        seq: chunked 序列
        num_tokens: 当前 chunk 的 token 数

    Returns:
        [seq_len, num_heads, head_dim] 最终 attention输出
    """
    pass
```

### 2.4 修改 select_blocks 接口

```python
def select_blocks(
    self,
    available_blocks: List[int],
    offload_engine: OffloadEngine,
    ctx: PolicyContext,
) -> List[int]:
    """
    选择要加载的 blocks。

    Args:
        available_blocks: 所有可用的 block IDs
        offload_engine: offload engine（为未来准备，当前可能不使用）
        ctx: policy context

    Returns:
        选择的 block IDs
    """
    pass
```

## Phase 3: 修改 FullPolicy，实现三个抽象方法

### 3.1 FullPolicy.compute_block_attention

直接调用 `flash_attn_with_lse`，处理 3D 输入。

### 3.2 FullPolicy.merge_attention_outputs

调用 `chunked_attention.merge_attention_outputs`。

### 3.3 FullPolicy.compute_prefill_attention

实现完整的 prefill 流程：
1. 获取 `cpu_block_table = kvcache_manager.get_prefilled_cpu_blocks(seq)`
2. 调用 `select_blocks(cpu_block_table, offload_engine, ctx)`
3. 遍历 blocks：
   - `offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)`
   - `offload_engine.wait_slot_layer(slot)`
   - `k, v = offload_engine.get_kv_for_slot(slot)`
   - 调用 `self.compute_block_attention(q, k, v, layer_id, scale, causal=False)`
   - 调用 `self.merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)`
4. 计算当前 chunk attention
5. 合并最终结果

### 需要移动的代码

从 `attention.py` 的 `_ring_buffer_pipeline_load` 和 `_sync_load_previous_chunks` 移动逻辑：
- slot 遍历逻辑
- offload_engine 调用
- 计算和合并逻辑

从 `attention.py` 的 `_chunked_prefill_attention` 移动逻辑：
- 当前 chunk 的 attention 计算
- 最终合并逻辑

## Phase 4: 修改 QuestPolicy

QuestPolicy 实现与 FullPolicy 类似，区别在于：
- `select_blocks` 返回 Top-K blocks
- 其他计算逻辑相同

## Phase 5: 修改 XAttentionBSAPolicy

当前 XAttentionBSAPolicy 只返回所有 blocks，修改后：
- `select_blocks` 当前返回所有 blocks
- `compute_block_attention` 与 FullPolicy 相同
- `merge_attention_outputs` 与 FullPolicy 相同
- `compute_prefill_attention` 与 FullPolicy 相同

未来可以实现稀疏计算。

## Phase 6: 修改 model_runner.py，默认创建 FullPolicy

### 6.1 当前创建 sparse policy 的逻辑

```python
# 当前：只有指定 sparse_policy_type 时才创建
if sparse_policy_type is not None:
    sparse_policy = create_sparse_policy(sparse_policy_type, **kwargs)
```

### 6.2 修改后

```python
# 默认创建 FullPolicy
if sparse_policy_type is None:
    sparse_policy_type = SparsePolicyType.FULL

sparse_policy = create_sparse_policy(sparse_policy_type, **kwargs)
```

### 6.3 位置

`model_runner.py` 中的 `allocate_kv_cache` 方法。

## Phase 7: 修改 attention.py，移除所有计算逻辑

### 7.1 _chunked_prefill_attention 简化

**当前（伪代码）**：
```python
# 获取 cpu_block_table
# 调用 select_blocks
# 调用 _ring_buffer_pipeline_load（包含计算逻辑）
# 计算当前 chunk（flash_attn）
# 合并结果（merge）
```

**修改后**：
```python
sparse_policy = kvcache_manager.sparse_policy
if sparse_policy is None:
    raise RuntimeError("sparse_policy is required for chunked prefill")

o = sparse_policy.compute_prefill_attention(
    q, k, v, self.layer_id, self.scale,
    offload_engine, current_chunk_idx, seq, num_tokens
)

# 直接返回，不需要合并（policy 内部已完成所有计算）
return o
```

### 7.2 删除的方法

删除以下方法（逻辑移到 policy 中）：
- `_ring_buffer_pipeline_load` - 逻辑移到 FullPolicy.compute_prefill_attention
- `_sync_load_previous_chunks` - 逻辑移到 FullPolicy.compute_prefill_attention

### 7.3 保留的方法

- `_decode_with_layer_pipeline` - decode 逻辑保持不变
- `_decode_ring_buffer_pipeline` - decode 逻辑保持不变

## Phase 8: 测试验证

- [ ] 运行 `test_needle.py --enable-offload` (FULL policy)
- [ ] 验证输出正确 (needle value: 7492)
- [ ] 验证性能无明显下降

## 关键文件清单

| 文件 | 修改内容 |
|------|----------|
| `nanovllm/kvcache/sparse/policy.py` | 添加三个抽象方法，修改 select_blocks 签名 |
| `nanovllm/kvcache/sparse/full_policy.py` | 实现三个抽象方法，移动计算逻辑 |
| `nanovllm/kvcache/sparse/quest.py` | 实现三个抽象方法 |
| `nanovllm/kvcache/sparse/xattn_bsa.py` | 实现三个抽象方法 |
| `nanovllm/engine/model_runner.py` | 默认创建 FullPolicy |
| `nanovllm/layers/attention.py` | 简化 _chunked_prefill_attention，删除计算方法 |

## Decisions Made

- **决策 1**: 三个方法都是抽象方法，强制所有 policy 实现
- **决策 2**: compute_prefill_attention 定义完整的 prefill 流程，是 policy 的主入口
- **决策 3**: attention.py 只调用 policy.compute_prefill_attention，零计算逻辑
- **决策 4**: chunked_prefill 检查 policy 是否存在，不存在则抛出错误
- **决策 5**: model_runner 默认创建 FullPolicy 作为兜底
- **决策 6**: _ring_buffer_pipeline_load 和 _sync_load_previous_chunks 删除，逻辑移到 policy

## Errors Encountered

（待记录）

## Status

**Currently in Phase 1** - 分析当前架构，理解所有计算逻辑的位置
