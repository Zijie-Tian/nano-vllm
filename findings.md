# Findings: CUDA Graph for Offload Mode

## Discovery 1: 为什么 Offload Mode 不使用 CUDA Graph

**位置**: `nanovllm/engine/model_runner.py:421`

```python
use_eager = is_prefill or self.enforce_eager or input_ids.size(0) > 512 or context.is_chunked_prefill
```

**原因**: `run_chunked_offload_decode()` 设置 `is_chunked_prefill=True`，强制使用 eager mode。

---

## Discovery 2: 当前 CUDA Graph 架构

**文件**: `model_runner.py:682-717`

```python
def capture_cudagraph(self):
    # 为不同 batch size 捕获完整 model forward
    for bs in [1, 2, 4, 8, 16, ...]:
        with torch.cuda.graph(graph):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
```

**特点**:
- 捕获完整的 `model()` 调用（包含所有层）
- 使用 graph pool 共享内存
- 只用于 decode（prefill 始终 eager）

---

## Discovery 3: Offload Decode 的 Attention 流程

**文件**: `nanovllm/kvcache/sparse/full_policy.py:304-379`

**Ring Buffer Pipeline**:
```
1. 预加载前 N 个 blocks 到 GPU slots
2. 对每个 block:
   a. wait_slot_layer()       # 等待 H2D
   b. get_kv_for_slot()       # 获取 KV
   c. flash_attn_with_lse()   # ⭐ 可 graph
   d. record_slot_compute_done()
   e. load_next_block()       # 启动下一个 H2D
   f. merge_attention_outputs() # ⭐ 可 graph（但动态）
```

**关键**: H2D 传输不能 graph，但 attention 计算可以。

---

## Discovery 4: 验证 Graph 复用可行性

**测试**: `tests/test_chunk_attention_graph_reuse.py`

**结论**:
- 只需 2 个 graph（causal + non-causal）
- 通过 `copy_()` 更新 static tensors
- 可复用于所有层和所有 chunk pairs

**测试结果**:
```
Layer 0: max_diff=3.91e-03 ✅
Layer 1: max_diff=7.81e-03 ✅
Layer 2: max_diff=3.91e-03 ✅
✅ PASSED
```

---

## Discovery 5: Chunk Size 和 Block Size 关系

**观察**:
- Prefilled blocks 的 KV size = `block_size`
- Decode buffer 的 KV size = `1` 到 `block_size`（动态）

**Graph 策略**:
- Prefilled blocks: 固定 size = block_size，适合 graph
- Decode buffer: 动态 size，建议保持 eager

---

## Discovery 6: 使用的 Triton 算子

**文件**: `nanovllm/ops/chunked_attention.py`

| 算子 | 功能 | 可 Graph |
|------|------|----------|
| `flash_attn_with_lse()` | Attention + LSE | ✅ |
| `merge_attention_outputs()` | 合并两个 attention 输出 | ✅ |

这两个算子是纯 GPU 计算，可以被 CUDA Graph 捕获。

---

## Discovery 7: 数据依赖分析

**Attention 输入**:
- `q`: 来自当前层的 QKV projection，shape 固定
- `k, v`: 来自 GPU slot（H2D 传输后），shape = [1, block_size, heads, dim]

**依赖链**:
```
H2D(block) → wait() → get_kv() → copy_to_static() → graph.replay() → clone_output()
```

**关键**: Graph 只封装 attention 计算，不包含数据传输。
