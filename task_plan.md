# Task Plan: CUDA Graph 优化 Offload Mode Decode

## 目标

为 nanovllm 的 CPU offload 模式添加 CUDA Graph 支持，加速 decode 阶段的计算。

## 问题分析

### Transformer 层的完整结构

```
Qwen3DecoderLayer.forward:
├── input_layernorm (RMSNorm)           # ✅ 纯 GPU
├── self_attn:
│   ├── qkv_proj (Linear)               # ✅ 纯 GPU
│   ├── q_norm, k_norm (RMSNorm)        # ✅ 纯 GPU
│   ├── rotary_emb                      # ✅ 纯 GPU
│   ├── attn._chunked_decode_attention: # ⚠️ 包含 CPU→GPU
│   │   ├── H2D transfer                # ❌ 不能 graph
│   │   ├── flash_attn_with_lse         # ✅ 可以 graph
│   │   └── merge                       # ✅ 纯 GPU
│   └── o_proj (Linear)                 # ✅ 纯 GPU
├── post_attention_layernorm            # ✅ 纯 GPU
└── mlp (FFN: gate, up, down)           # ✅ 纯 GPU
```

**核心问题**：H2D 传输被嵌在 attention 中间，打断了整层的 graph 捕获。

### 可能的方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 分段 Graph | 将层拆分为 pre/post attention 两段 | 覆盖面广 | 改动大，需拆分层执行 |
| B. 只 Graph Attention | 只优化 flash_attn_with_lse | 改动小 | 优化效果有限 |
| C. 重构执行流程 | 完全重写 model forward | 最优效果 | 工作量巨大 |

### 推荐：方案 A（分段 Graph）

将每层拆分为两个 graph：
1. **pre_attention_graph**: `norm → qkv_proj → q/k_norm → rotary`
2. **post_attention_graph**: `o_proj → norm → FFN`

中间的 `_chunked_decode_attention` 保持 eager（包含 H2D），但内部的 `flash_attn_with_lse` 使用 graph。

---

## 当前状态分析

### 现有 CUDA Graph 实现

**文件**: `nanovllm/engine/model_runner.py`

| 方法 | 行号 | 功能 |
|------|------|------|
| `capture_cudagraph()` | 682-717 | 为不同 batch size 捕获完整 model forward |
| `run_model()` | 415-436 | 决定使用 eager 还是 graph replay |

**关键逻辑** (`run_model`):
```python
use_eager = is_prefill or self.enforce_eager or input_ids.size(0) > 512 or context.is_chunked_prefill
```

**问题**: `run_chunked_offload_decode` 设置 `is_chunked_prefill=True`，导致**永远使用 eager mode**。

### Offload Decode 流程

**文件**: `nanovllm/kvcache/sparse/full_policy.py`

`_decode_ring_buffer_pipeline()` (L304-379):
```
for block in cpu_blocks:
    1. wait_slot_layer(slot)           # 等待 H2D 完成
    2. k, v = get_kv_for_slot(slot)    # 获取 KV
    3. o, lse = flash_attn_with_lse()  # ⭐ 纯 GPU 计算
    4. record_slot_compute_done(slot)  # 标记计算完成
    5. load_next_block()               # 启动下一个 H2D
    6. merge_attention_outputs()       # ⭐ 纯 GPU 计算
```

**可 Graph 化的部分**:
- `flash_attn_with_lse()` - 纯 GPU 计算
- 不可 Graph 化: H2D 传输、动态 merge

## 验证结果

**测试文件**: `tests/test_chunk_attention_graph_reuse.py`

| 测试 | 结果 |
|------|------|
| 2 个 Graph 复用于所有层和所有 chunk | ✅ PASSED |
| copy_() 更新 static tensors | ✅ 有效 |
| Eager merge | ✅ 用户已接受 |

**结论**: 只需 2 个 graph（causal + non-causal），通过 copy_() 复用。

---

## 修改计划（方案 A：分段 Graph）

### 架构设计

```
每层执行流程（Offload Decode）:
┌─────────────────────────────────────────────────────────────┐
│  PRE-ATTENTION GRAPH (可复用于所有层)                        │
│  input_layernorm → qkv_proj → q/k_norm → rotary → split Q  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  CHUNKED ATTENTION (Eager + 部分 Graph)                     │
│  for block in cpu_blocks:                                   │
│      H2D transfer (eager)                                   │
│      flash_attn_with_lse (GRAPH - 2个可复用)                │
│      merge (eager)                                          │
│  decode_buffer attention (eager)                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  POST-ATTENTION GRAPH (可复用于所有层)                       │
│  o_proj → post_layernorm → gate_proj → up_proj → SiLU      │
│  → down_proj → residual                                     │
└─────────────────────────────────────────────────────────────┘
```

**总共需要的 Graph 数量**:
- 1 个 pre_attention_graph（所有层复用）
- 2 个 attention_graph（causal + non-causal，所有层复用）
- 1 个 post_attention_graph（所有层复用）
- **总计: 4 个 graph**

---

### Phase 1: 拆分 DecoderLayer 执行

**目标**: 将 `Qwen3DecoderLayer.forward` 拆分为可独立调用的三段

**修改文件**: `nanovllm/models/qwen3.py`

**新增方法**:
```python
class Qwen3DecoderLayer:
    def forward_pre_attention(self, positions, hidden_states, residual):
        """Pre-attention: norm → qkv → rotary → 返回 q, k, v"""
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        qkv = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.self_attn.q_norm(q)
        k = self.self_attn.k_norm(k)
        q, k = self.self_attn.rotary_emb(positions, q, k)
        return q, k, v, hidden_states, residual

    def forward_post_attention(self, attn_output, hidden_states, residual):
        """Post-attention: o_proj → norm → FFN"""
        output = self.self_attn.o_proj(attn_output.flatten(1, -1))
        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**状态**: `pending`

---

### Phase 2: 捕获 Pre/Post Attention Graph

**目标**: 捕获 pre_attention 和 post_attention 的 graph

**修改文件**: `nanovllm/engine/model_runner.py`

**新增方法**: `capture_offload_layer_graphs()`

```python
def capture_offload_layer_graphs(self):
    """捕获 offload mode 的 layer graphs"""
    # 获取任意一层作为模板（所有层结构相同）
    layer = self.model.model.layers[0]

    # Static tensors
    static_hidden = torch.zeros(1, self.hidden_size, ...)
    static_residual = torch.zeros(1, self.hidden_size, ...)
    static_positions = torch.zeros(1, ...)

    # Pre-attention graph
    self.pre_attn_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.pre_attn_graph):
        static_q, static_k, static_v, _, _ = layer.forward_pre_attention(
            static_positions, static_hidden, static_residual
        )

    # Post-attention graph
    self.post_attn_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.post_attn_graph):
        _, _ = layer.forward_post_attention(
            static_attn_output, static_hidden, static_residual
        )
```

**状态**: `pending`

---

### Phase 3: 捕获 Attention Graph

**目标**: 捕获 2 个 attention graph（causal + non-causal）

**修改文件**: `nanovllm/kvcache/offload_engine.py`

```python
class OffloadEngine:
    def capture_attention_graphs(self):
        """捕获 attention graphs（复用于所有层）"""
        self.attn_graph_causal = self._capture_attn_graph(causal=True)
        self.attn_graph_non_causal = self._capture_attn_graph(causal=False)

    def _capture_attn_graph(self, causal: bool):
        static_q = torch.zeros(1, 1, num_heads, head_dim, ...)
        static_k = torch.zeros(1, block_size, num_kv_heads, head_dim, ...)
        static_v = torch.zeros(1, block_size, num_kv_heads, head_dim, ...)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, lse = flash_attn_with_lse(static_q, static_k, static_v,
                                              self.scale, causal)
        return AttentionGraph(graph, static_q, static_k, static_v, output, lse)
```

**状态**: `pending`

---

### Phase 4: 修改 Offload Decode 执行流程

**目标**: 使用 graph replay 执行 offload decode

**修改文件**: `nanovllm/engine/model_runner.py`

**修改方法**: `run_chunked_offload_decode()`

```python
def run_chunked_offload_decode_with_graph(self, seqs):
    """使用 graph 加速的 offload decode"""
    seq = seqs[0]

    # 准备输入
    input_ids = torch.tensor([seq.last_token], ...)
    positions = torch.tensor([len(seq) - 1], ...)

    # Embedding
    hidden_states = self.model.model.embed_tokens(input_ids)
    residual = None

    for layer_id, layer in enumerate(self.model.model.layers):
        # Phase 1: Pre-attention (GRAPH)
        self.pre_attn_vars["hidden"].copy_(hidden_states)
        self.pre_attn_vars["residual"].copy_(residual) if residual else None
        self.pre_attn_vars["positions"].copy_(positions)
        self.pre_attn_graph.replay()
        q = self.pre_attn_vars["q"].clone()
        k = self.pre_attn_vars["k"].clone()
        v = self.pre_attn_vars["v"].clone()

        # Phase 2: Chunked Attention (Eager + Graph)
        attn_output = self._chunked_attention_with_graph(q, k, v, layer_id, ...)

        # Phase 3: Post-attention (GRAPH)
        self.post_attn_vars["attn_output"].copy_(attn_output)
        self.post_attn_graph.replay()
        hidden_states = self.post_attn_vars["hidden"].clone()
        residual = self.post_attn_vars["residual"].clone()

    # LM head
    logits = self.model.compute_logits(hidden_states)
    return logits
```

**状态**: `pending`

---

### Phase 5: 修改 Ring Buffer Pipeline

**目标**: 在 attention 内部使用 graph

**修改文件**: `nanovllm/kvcache/sparse/full_policy.py`

**修改**: `_decode_ring_buffer_pipeline()` 中的 `flash_attn_with_lse` 调用

```python
# 当前：eager
prev_o, prev_lse = flash_attn_with_lse(q, k, v, scale, causal=False)

# 修改为：graph replay
graph = offload_engine.attn_graph_non_causal
graph.static_q.copy_(q)
graph.static_k.copy_(k)
graph.static_v.copy_(v)
graph.graph.replay()
prev_o = graph.static_output.clone()
prev_lse = graph.static_lse.clone()
```

**状态**: `pending`

---

### Phase 6: 添加配置开关

**修改文件**: `nanovllm/config.py`

```python
enable_offload_graph: bool = True  # 默认启用
```

**状态**: `pending`

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `nanovllm/engine/model_runner.py` | 新增方法 | `capture_offload_attention_graph()` |
| `nanovllm/kvcache/offload_engine.py` | 新增属性+方法 | Graph 存储和访问 |
| `nanovllm/kvcache/sparse/full_policy.py` | 修改方法 | 使用 graph replay |
| `nanovllm/config.py` | 新增配置 | `enable_offload_graph` |

---

## 风险和注意事项

1. **Graph 捕获时机**: 需要在 KV cache 分配后、第一次 decode 前捕获
2. **Chunk size 匹配**: Graph 的 chunk_size 必须和 block_size 一致
3. **多 GPU**: Graph 需要在每个 GPU 上分别捕获
4. **内存**: 2 个 graph 的额外内存开销很小

---

## 测试计划

1. **单元测试**: 验证 graph replay 结果正确
2. **集成测试**: 运行 `test_needle.py --enable-offload --input-len 32768`
3. **性能测试**: 对比 eager vs graph 的 decode 延迟

---

## 预期收益

- Decode 阶段 attention 计算加速（减少 kernel launch overhead）
- 与现有 ring buffer pipeline 兼容
- 内存开销极小（只有 2 个额外 graph）
