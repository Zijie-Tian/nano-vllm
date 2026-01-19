# Task Plan: Sparse Policy 架构重构 v4 (FullPolicy Only)

## Goal

将 chunked prefill 的 attention 计算逻辑完全从 `attention.py` 移到 `SparsePolicy` 内部。attention.py 只负责调用 policy，不包含任何计算逻辑。

**范围**: 仅实现 FullPolicy，暂不涉及 QuestPolicy 和 XAttentionBSAPolicy。Decode 阶段不处理。

## 当前代码状态（重要发现）

**`FullPolicy.compute_prefill_attention` 已经实现了完整的 prefill 流程！**

但 `attention.py` 没有调用它，而是：
- 调用 `sparse_policy.select_blocks()` 仅做 block 筛选
- 自己实现 `_ring_buffer_pipeline_load` 和 `_sync_load_previous_chunks`
- 自己调用 `flash_attn_with_lse` 和 `merge_attention_outputs`

**结论**：当前代码有冗余，同样的逻辑在两个地方实现。

## 核心设计原则

1. **Policy 内部完成所有 prefill 计算**：包括 block 加载、attention 计算和结果合并
2. **select_blocks 传入 offload_engine**：其他策略（Quest/XAttn）可能需要加载 KV 来判断
3. **统一方法命名**：使用 `compute_chunked_attention`（不是 `compute_prefill_attention`）
4. **chunked_prefill 强制 policy 存在**：没有 policy 则报错
5. **attention.py 零计算逻辑**：`_chunked_prefill_attention` 只调用 policy

## 目标架构

```
attention.py (_chunked_prefill_attention):
  检查 sparse_policy 是否存在
    ↓
  调用 sparse_policy.compute_chunked_attention(q, k, v, ...)
    ↓
  处理 async offload
    ↓
  返回最终输出（不包含任何 attention 计算逻辑）

SparsePolicy.compute_chunked_attention():
  1. 获取 cpu_block_table
  2. 调用 select_blocks(blocks, offload_engine, ctx) → 筛选 blocks
  3. 加载 blocks 并计算 attention（pipeline 或 sync）
  4. 计算当前 chunk attention（causal）
  5. 合并所有结果
  6. 返回 final_output
```

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **决策 1** | `compute_chunked_attention` 是唯一的抽象方法，定义完整 prefill 流程 |
| **决策 2** | 不添加 `compute_block_attention` 和 `merge_attention_outputs` 抽象方法（过度设计） |
| **决策 3** | `select_blocks` 接收 `offload_engine` 参数（其他策略需要） |
| **决策 4** | attention.py 的 `_chunked_prefill_attention` 不包含任何 flashattn 或 merge 调用 |
| **决策 5** | Decode 阶段不处理，保持现有逻辑 |
| **决策 6** | async offload 逻辑保留在 attention.py（不移入 policy） |
| **决策 7** | Phase 4 需要添加 debug 输出验证执行路径 |

## Phases

- [x] Phase 1: 分析当前架构 ✅ 已完成
- [ ] Phase 2: 修改 SparsePolicy 基类
- [ ] Phase 3: 修改 FullPolicy
- [ ] Phase 4: 验证执行路径（添加 debug 输出）
- [ ] Phase 5: 修改 attention.py
- [ ] Phase 6: 测试验证

## Phase 1: 分析当前架构 ✅ 已完成

### 当前 attention.py 中包含的计算逻辑（需要移除）

1. `_ring_buffer_pipeline_load` 方法：直接调用 flashattn 和 merge
2. `_sync_load_previous_chunks` 方法：直接调用 flashattn 和 merge
3. `_chunked_prefill_attention` 方法：
   - 调用上述两个方法
   - 计算当前 chunk（flash_attn）
   - 合并结果（merge）

### 当前 FullPolicy 已实现的功能

`full_policy.py:40-162` 的 `compute_prefill_attention` 已实现：
- ring buffer pipeline 加载
- sync 加载 fallback
- 当前 chunk attention 计算
- 结果合并

**只需重命名为 `compute_chunked_attention` 并微调接口。**

## Phase 2: 修改 SparsePolicy 基类

### 2.1 修改 select_blocks 接口

```python
@abstractmethod
def select_blocks(
    self,
    available_blocks: List[int],
    offload_engine: "OffloadEngine",  # 新增参数
    ctx: PolicyContext,
) -> List[int]:
    """
    选择要加载的 blocks。

    Args:
        available_blocks: 所有可用的 block IDs
        offload_engine: offload engine（其他策略可能需要加载 KV 来判断）
        ctx: policy context

    Returns:
        选择的 block IDs
    """
    pass
```

### 2.2 添加 compute_chunked_attention 抽象方法

```python
@abstractmethod
def compute_chunked_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    current_chunk_idx: int,
    seq: "ChunkedSequence",
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
        q: [seq_len, num_heads, head_dim] 当前 chunk 的 query
        k, v: [seq_len, num_kv_heads, head_dim] 当前 chunk 的 KV（已写入 prefill buffer）
        layer_id: 层索引
        softmax_scale: softmax 缩放因子
        offload_engine: offload engine
        current_chunk_idx: 当前 chunk 索引
        seq: chunked 序列
        num_tokens: 当前 chunk 的 token 数

    Returns:
        [seq_len, num_heads, head_dim] 最终 attention 输出
    """
    pass
```

## Phase 3: 修改 FullPolicy

### 3.1 重命名方法

将 `compute_prefill_attention` 重命名为 `compute_chunked_attention`。

### 3.2 修改 select_blocks 签名

```python
def select_blocks(
    self,
    available_blocks: List[int],
    offload_engine: "OffloadEngine",  # 新增参数（不使用）
    ctx: PolicyContext,
) -> List[int]:
    """Return all blocks - no sparsity."""
    return available_blocks
```

### 3.3 验证 compute_chunked_attention 实现

当前 `compute_prefill_attention` 已实现完整逻辑，确认：
- [x] 获取 cpu_block_table
- [x] ring buffer pipeline 加载
- [x] sync 加载 fallback
- [x] 当前 chunk attention 计算
- [x] 结果合并

**注意**：当前实现没有调用 `select_blocks`，需要添加。

## Phase 4: 验证执行路径（添加 debug 输出）

### 4.1 验证目标

确认代码修改后，执行路径正确：

| 检查点 | 位置 | 预期行为 |
|--------|------|----------|
| **Policy 创建** | `kvcache/__init__.py` | FullAttentionPolicy 被创建 |
| **Policy 调用** | `attention.py` | `_chunked_prefill_attention` 调用 `sparse_policy.compute_chunked_attention` |
| **select_blocks 调用** | `full_policy.py` | `compute_chunked_attention` 内部调用 `select_blocks` |
| **旧方法未调用** | `attention.py` | `_ring_buffer_pipeline_load` 和 `_sync_load_previous_chunks` 不再被调用 |

### 4.2 添加 debug 输出位置

**位置 1: `kvcache/__init__.py` - policy 创建时**
```python
sparse_policy = create_sparse_policy(sparse_policy_type, **policy_kwargs)
logger.info(f"[DEBUG] Created sparse policy: {sparse_policy}")
```

**位置 2: `attention.py` - 调用 policy 时**
```python
# 在 _chunked_prefill_attention 中
logger.debug(f"[DEBUG] Calling sparse_policy.compute_chunked_attention, "
             f"policy={sparse_policy}, layer={self.layer_id}, chunk={current_chunk_idx}")
```

**位置 3: `full_policy.py` - compute_chunked_attention 入口**
```python
def compute_chunked_attention(self, ...):
    logger.debug(f"[DEBUG] FullPolicy.compute_chunked_attention called, "
                 f"layer={layer_id}, chunk={current_chunk_idx}, num_tokens={num_tokens}")
    # ... 实现
```

**位置 4: `full_policy.py` - select_blocks 调用**
```python
# 在 compute_chunked_attention 内部
selected_blocks = self.select_blocks(cpu_block_table, offload_engine, policy_ctx)
logger.debug(f"[DEBUG] select_blocks: input={len(cpu_block_table)} blocks, "
             f"output={len(selected_blocks)} blocks")
```

### 4.3 验证方法

运行测试并检查日志输出：
```bash
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_needle.py --model <model_path> --enable-offload 2>&1 | grep DEBUG
```

预期输出：
```
[DEBUG] Created sparse policy: FullAttentionPolicy()
[DEBUG] Calling sparse_policy.compute_chunked_attention, policy=FullAttentionPolicy(), layer=0, chunk=0
[DEBUG] FullPolicy.compute_chunked_attention called, layer=0, chunk=0, num_tokens=...
[DEBUG] select_blocks: input=0 blocks, output=0 blocks
[DEBUG] Calling sparse_policy.compute_chunked_attention, policy=FullAttentionPolicy(), layer=0, chunk=1
[DEBUG] FullPolicy.compute_chunked_attention called, layer=0, chunk=1, num_tokens=...
[DEBUG] select_blocks: input=1 blocks, output=1 blocks
...
```

### 4.4 清理 debug 输出

验证完成后，将 debug 级别的日志改为更低级别（如 `logger.debug`），或通过环境变量控制：
```python
if os.environ.get('NANOVLLM_DEBUG_POLICY'):
    logger.info(f"[DEBUG] ...")
```

## Phase 5: 修改 attention.py

### 5.1 简化 _chunked_prefill_attention

**修改后**：
```python
def _chunked_prefill_attention(self, q, k, v, context):
    kvcache_manager = context.kvcache_manager
    seq = context.chunked_seq
    offload_engine = kvcache_manager.offload_engine
    current_chunk_idx = context.current_chunk_idx
    num_tokens = k.shape[0]

    # 获取 sparse policy
    sparse_policy = kvcache_manager.sparse_policy
    if sparse_policy is None:
        raise RuntimeError("sparse_policy is required for chunked prefill")

    # [DEBUG] 验证执行路径
    logger.debug(f"[DEBUG] Calling sparse_policy.compute_chunked_attention, "
                 f"policy={sparse_policy}, layer={self.layer_id}, chunk={current_chunk_idx}")

    # 调用 policy 计算 attention（所有计算逻辑在 policy 内部）
    final_o = sparse_policy.compute_chunked_attention(
        q, k, v,
        self.layer_id,
        self.scale,
        offload_engine,
        current_chunk_idx,
        seq,
        num_tokens,
    )

    # Per-layer ASYNC offload（保留在 attention.py）
    if offload_engine is not None and seq is not None:
        cpu_block_ids, _ = kvcache_manager.get_all_cpu_blocks(seq)
        if current_chunk_idx < len(cpu_block_ids):
            cpu_block_id = cpu_block_ids[current_chunk_idx]
            offload_engine.offload_prefill_buffer_async(
                self.layer_id, cpu_block_id, num_tokens
            )

    return final_o
```

### 5.2 删除的方法

删除以下方法（逻辑已移到 FullPolicy）：
- `_ring_buffer_pipeline_load`
- `_sync_load_previous_chunks`

### 5.3 保留的方法

Decode 相关方法保持不变：
- `_chunked_decode_attention`
- `_decode_with_layer_pipeline`
- `_decode_ring_buffer_pipeline`

## Phase 6: 测试验证

### 6.1 功能测试

- [ ] 运行 `test_needle.py --enable-offload` (FULL policy)
- [ ] 验证输出正确（needle value 匹配）
- [ ] 检查 debug 日志确认执行路径正确

### 6.2 性能测试

- [ ] 对比重构前后的 prefill 延迟
- [ ] 验证性能无明显下降（< 5% 回归）

### 6.3 回归测试

- [ ] 验证 decode 阶段不受影响
- [ ] 验证非 offload 模式不受影响（如果适用）

## 关键文件清单

| 文件 | 修改内容 |
|------|----------|
| `nanovllm/kvcache/sparse/policy.py` | 添加 `compute_chunked_attention` 抽象方法，修改 `select_blocks` 签名 |
| `nanovllm/kvcache/sparse/full_policy.py` | 重命名方法，修改 `select_blocks` 签名，添加 `select_blocks` 调用，添加 debug 输出 |
| `nanovllm/layers/attention.py` | 简化 `_chunked_prefill_attention`，删除 `_ring_buffer_pipeline_load` 和 `_sync_load_previous_chunks`，添加 debug 输出 |
| `nanovllm/kvcache/__init__.py` | 添加 policy 创建的 debug 输出 |

## Decisions Made

- **决策 1**: 只添加一个抽象方法 `compute_chunked_attention`（不添加 `compute_block_attention` 和 `merge_attention_outputs`）
- **决策 2**: `select_blocks` 接收 `offload_engine` 参数
- **决策 3**: 统一使用 `compute_chunked_attention` 命名
- **决策 4**: Decode 阶段不处理
- **决策 5**: async offload 逻辑保留在 attention.py（不移入 policy）
- **决策 6**: Phase 4 添加 debug 输出验证执行路径，验证完成后可降级或移除

## Errors Encountered

（待记录）

## Status

**Planning Complete** - v4 计划已完成，包含执行路径验证步骤
