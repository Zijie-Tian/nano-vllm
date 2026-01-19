# SparsePolicy 重构测试报告

## 任务概述

根据 task_plan.md 的要求，对 nanovllm 的 SparsePolicy 架构进行重构（v4 版本），将 chunked prefill attention 计算逻辑从 attention.py 完全迁移到 SparsePolicy。

## 修改范围

仅针对 FullPolicy，不涉及 QuestPolicy 或 XAttentionBSAPolicy，不修改 decode 阶段逻辑。

## 完成的修改

### 1. policy.py (SparsePolicy 基类)

- 添加 TYPE_CHECKING imports: `OffloadEngine`, `KVCacheManager`, `Sequence`
- 修改 `select_blocks` 签名：添加 `offload_engine` 参数
- 添加 `compute_chunked_attention` 抽象方法，参数包括：
  - `q, k, v`: 张量
  - `layer_id`: 层索引
  - `softmax_scale`: softmax 缩放因子
  - `offload_engine`: OffloadEngine 实例
  - `kvcache_manager`: KVCacheManager 实例
  - `current_chunk_idx`: 当前 chunk 索引
  - `seq`: Sequence 对象
  - `num_tokens`: 当前 chunk 的 token 数

### 2. full_policy.py (FullAttentionPolicy)

- 更新 TYPE_CHECKING imports
- `select_blocks` 方法签名添加 `offload_engine` 参数
- 重命名 `compute_prefill_attention` → `compute_chunked_attention`
- 添加 `kvcache_manager` 参数，替换所有 `seq.kvcache_manager` 引用
- 添加 debug 日志输出

### 3. attention.py

- 简化 `_chunked_prefill_attention` 方法：
  - 删除所有 `flash_attn_*` 调用
  - 删除所有 `merge_attention_outputs` 调用
  - 仅保留委托调用 `sparse_policy.compute_chunked_attention()`
- 删除冗余方法：`_sync_load_previous_chunks`, `_ring_buffer_pipeline_load`
- decode 路径的 `select_blocks` 调用添加 `offload_engine` 参数

## 验收标准检查

| 标准 | 状态 | 说明 |
|------|------|------|
| test_needle.py --enable-offload 通过 | ✅ | 测试输出 PASSED |
| attention.py chunked prefill path 无 flash_attn_* 调用 | ✅ | `_chunked_prefill_attention` 方法（169-230行）内无直接 flash_attn 调用 |
| attention.py chunked prefill path 无 merge_attention_outputs 调用 | ✅ | 同上 |
| 所有 KV 通信通过 offload_engine 方法 | ✅ | 全部通过 `offload_engine.load_to_slot_layer`, `get_kv_for_slot`, `get_prefill_buffer_slice` |

## 测试结果

```
============================================================
Needle-in-Haystack Test
============================================================
Model: /home/zijie/models/Llama-3.1-8B-Instruct
Max model len: 131072
Input length: 8192
Block size: 1024
Needle position: 50%
Needle value: 7492
CPU offload: True
Sparse policy: FULL
============================================================

[NeedleTest] Target: 8192, Actual: 8213 tokens (diff=21)
Expected: 7492
Output: 7492<|eot_id|>...
Status: PASSED
============================================================

test_needle: PASSED
```

## 性能指标

- Prefill: 3527 tok/s
- Decode: 11 tok/s
- TTFT: 2329.29 ms
- TPOT: 655.38 ms

## 架构变更总结

**重构前**:
```
attention.py::_chunked_prefill_attention()
  ├── 获取 cpu_block_table
  ├── 调用 sparse_policy.select_blocks()
  ├── 直接调用 flash_attn_with_lse + merge_attention_outputs
  └── 返回结果
```

**重构后**:
```
attention.py::_chunked_prefill_attention()
  ├── 获取 context 信息
  ├── 调用 sparse_policy.compute_chunked_attention()  # 委托全部计算
  └── 返回结果

sparse_policy.compute_chunked_attention()  # 在 FullPolicy 中
  ├── 获取 cpu_block_table
  ├── 调用 self.select_blocks()
  ├── 加载并计算历史 KV attention
  ├── 计算当前 chunk attention (causal)
  ├── 合并所有结果
  └── 返回最终输出
```

## 结论

SparsePolicy 架构 v4 重构成功完成。所有验收标准均已满足，测试通过。
