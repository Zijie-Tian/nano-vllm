# XAttention BSA Policy 设计文档

本文档描述 `XAttentionBSAPolicy` 的设计和实现，这是一个基于 XAttention 算法的稀疏注意力策略，用于 CPU offload 模式下的 chunked prefill。

## 概述

`XAttentionBSAPolicy` 实现了基于 XAttention 的块级稀疏注意力选择。核心思想是：

1. **估计阶段**：使用 XAttention kernels 快速估计每个 KV block 的重要性
2. **选择阶段**：基于阈值和 majority voting 选择重要的 blocks
3. **计算阶段**：只加载选中的 blocks 进行 attention 计算

```
┌─────────────────────────────────────────────────────────────┐
│                    XAttention BSA Policy                     │
├─────────────────────────────────────────────────────────────┤
│  select_blocks()                                             │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ Load K      │──>│ flat_group_gemm  │──>│ softmax_fuse │  │
│  │ blocks      │   │ _fuse_reshape    │   │ _block_sum   │  │
│  └─────────────┘   └──────────────────┘   └──────────────┘  │
│         │                   │                    │           │
│         v                   v                    v           │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ K: [B,H,L,D]│   │ attn_scores:     │   │ block_sums:  │  │
│  │             │   │ [B,H,Q/s,K/s]    │   │ [B,H,Qb,Kb]  │  │
│  └─────────────┘   └──────────────────┘   └──────────────┘  │
│                                                  │           │
│                           ┌──────────────────────┘           │
│                           v                                  │
│                    ┌──────────────┐                          │
│                    │find_blocks   │                          │
│                    │_chunked      │                          │
│                    └──────────────┘                          │
│                           │                                  │
│                           v                                  │
│                    ┌──────────────┐                          │
│                    │ GQA-aware    │                          │
│                    │ aggregation  │                          │
│                    │ + majority   │                          │
│                    │ voting       │                          │
│                    └──────────────┘                          │
│                           │                                  │
│                           v                                  │
│                    selected_block_ids                        │
├─────────────────────────────────────────────────────────────┤
│  compute_chunked_prefill()                                   │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ Ring buffer │──>│ flash_attn_      │──>│ merge_       │  │
│  │ pipeline    │   │ with_lse         │   │ attention    │  │
│  └─────────────┘   └──────────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 文件位置

**主文件**: `nanovllm/kvcache/sparse/xattn_bsa.py`

**依赖的 XAttention kernels**: `nanovllm/ops/xattn.py`
- `flat_group_gemm_fuse_reshape`: 计算 stride reshape 后的 attention scores
- `softmax_fuse_block_sum`: 对 attention scores 做 softmax 后按 block 求和
- `find_blocks_chunked`: 基于阈值选择 blocks

---

## 核心算法

### 1. select_blocks: 块选择算法

```python
def select_blocks(self, available_blocks, offload_engine, ctx) -> List[int]:
```

#### Step 1: 加载 K blocks 并计算 attention scores

对每个 CPU block，加载 K 到 GPU 并使用 `flat_group_gemm_fuse_reshape` 计算：

```python
for cpu_block_id in available_blocks:
    # 加载 K block: [1, block_size, num_kv_heads, head_dim]
    offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
    k_block, _ = offload_engine.get_kv_for_slot(slot)

    # 转换为 [batch, heads, k_len, head_dim]
    K_chunk = k_block.transpose(1, 2)

    # GQA: 扩展 K heads 匹配 Q heads
    if num_heads != num_kv_heads:
        K_chunk = K_chunk.repeat_interleave(num_groups, dim=1)

    # 计算 attention scores
    attn_chunk = flat_group_gemm_fuse_reshape(Q, K_chunk, stride, ...)
    attn_scores_list.append(attn_chunk)

# 拼接所有 K chunks: [1, heads, q_reshaped_len, total_k_reshaped_len]
attn_scores = torch.cat(attn_scores_list, dim=-1)
```

#### Step 2: 聚合到 block 级别

使用 `softmax_fuse_block_sum` 将 attention scores 聚合到 block 级别：

```python
# reshaped_block_size = block_size / stride = 1024 / 8 = 128
block_sums = softmax_fuse_block_sum(
    attn_scores,
    reshaped_block_size,  # 1:1 对应 CPU blocks
    segment_size,
    chunk_start=0,
    chunk_end=q_reshaped_len,
    real_q_len=q_reshaped_len,
    scale=scale,
    is_causal=False,
)
# block_sums: [batch, heads, q_blocks, k_blocks]
```

**关键点**: `reshaped_block_size` 必须与 CPU block 对齐，确保输出的 `k_blocks` 维度 1:1 对应 `available_blocks`。

#### Step 3: 阈值选择

使用 `find_blocks_chunked` 基于累积注意力阈值选择 blocks：

```python
mask = find_blocks_chunked(
    block_sums,
    current_index=0,
    threshold=self.threshold,  # e.g., 0.95
    num_to_choose=None,
    decoding=False,
    mode="prefill",
    causal=False,
)
# mask: [batch, num_heads, q_blocks, k_blocks] - boolean
```

#### Step 4: GQA-aware 聚合 + Majority Voting

```python
# GQA: 在同一个 KV head group 内，任一 Q head 选择即选择
if num_groups > 1:
    mask_gqa = mask.view(batch_size, num_kv_heads, num_groups, q_blocks, k_blocks)
    mask_per_kv_head = mask_gqa.any(dim=2)  # [batch, num_kv_heads, q_blocks, k_blocks]

# Majority voting: 跨 KV heads 和 q_blocks 投票
vote_count = mask_per_kv_head[0].float().sum(dim=0).sum(dim=0)  # [k_blocks]
total_votes = num_kv_heads * q_blocks
vote_ratio = vote_count / total_votes

# 选择 >50% 投票的 blocks
vote_threshold = 0.5
block_selected = vote_ratio > vote_threshold
selected_block_ids = [available_blocks[i] for i, sel in enumerate(block_selected.tolist()) if sel]

# 安全措施: 始终包含第一个 (sink) 和最后一个 block
if available_blocks[0] not in selected_block_ids:
    selected_block_ids.insert(0, available_blocks[0])
if available_blocks[-1] not in selected_block_ids:
    selected_block_ids.append(available_blocks[-1])
```

**为什么使用 Majority Voting?**

| 聚合方式 | 问题 |
|---------|------|
| `any()` 跨所有 heads | 密度接近 100%，失去稀疏性 |
| `all()` | 太激进，可能丢失重要 blocks |
| **Majority voting (>50%)** | 平衡稀疏性和准确性 |

实验结果显示：
- 每 head 密度: 20-35%
- `any()` 聚合后: ~100%
- **Majority voting 后: ~45%**

---

### 2. compute_chunked_prefill: 注意力计算

复用 `FullAttentionPolicy` 的 ring buffer pipeline 实现：

```python
def compute_chunked_prefill(self, q, k, v, layer_id, softmax_scale,
                            offload_engine, kvcache_manager,
                            current_chunk_idx, seq, num_tokens,
                            selected_blocks) -> torch.Tensor:
```

#### 计算流程

1. **加载历史 blocks** (使用 selected_blocks):
   ```python
   for block_idx in range(num_blocks):
       # Ring buffer pipeline: load -> wait -> compute -> next
       offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
       offload_engine.wait_slot_layer(slot)

       prev_k, prev_v = offload_engine.get_kv_for_slot(slot)
       prev_o, prev_lse = flash_attn_with_lse(q, prev_k, prev_v, causal=False)

       o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, prev_o, prev_lse)
   ```

2. **计算当前 chunk** (causal mask):
   ```python
   k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
   current_o, current_lse = flash_attn_with_lse(q, k_curr, v_curr, causal=True)
   ```

3. **合并结果**:
   ```python
   final_o, _ = merge_attention_outputs(o_acc, lse_acc, current_o, current_lse)
   ```

---

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 0.95 | 累积注意力阈值 (tau)，越高越保守 |
| `stride` | 8 | XAttention stride reshape 参数 |
| `chunk_size` | 16384 | 估计时的处理 chunk size |
| `block_size` | 128 | BSA block size (固定值) |

### 使用方式

```python
# 在 config 中设置
config.sparse_policy = SparsePolicyType.XATTN_BSA
config.sparse_threshold = 0.95

# 或通过命令行
python tests/test_needle.py \
    --enable-offload \
    --enable-xattn-bsa \
    --sparse-threshold 9  # 会被除以 10 变为 0.9
```

---

## 性能特性

| 特性 | 说明 |
|------|------|
| **Prefill 支持** | ✅ 完整支持 |
| **Decode 支持** | ❌ 不支持（使用 FullAttentionPolicy） |
| **稀疏度** | ~45-55%（threshold=0.95，majority voting） |
| **准确性** | RULER NIAH 100% 通过 |

### 限制

1. **Decode 不支持**: XAttention 估计需要足够长的 Q 序列，单 token decode 不适用
2. **估计开销**: `select_blocks` 需要加载所有 K blocks 进行估计
3. **Triton 对齐**: Q/K 长度必须满足 `stride * BLOCK_M/N` 对齐要求

---

## 与其他 Policy 的对比

| Policy | select_blocks | 稀疏度 | Decode 支持 |
|--------|--------------|--------|-------------|
| FullAttentionPolicy | 返回所有 blocks | 0% | ✅ |
| QuestPolicy | 基于 min/max key | ~50% | ✅ |
| **XAttentionBSAPolicy** | XAttention + majority voting | ~45-55% | ❌ |

---

## 测试验证

```bash
# Needle test (32K)
CUDA_VISIBLE_DEVICES=0 python tests/test_needle.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --enable-xattn-bsa \
    --input-len 32768

# RULER benchmark
CUDA_VISIBLE_DEVICES=0 python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sparse-policy XATTN_BSA \
    --sparse-threshold 0.95 \
    --data-dir tests/data/ruler_niah
```

---

## 相关文档

- [`docs/xattention_algorithm_guide.md`](xattention_algorithm_guide.md): XAttention 算法详解
- [`docs/xattn_kernels_guide.md`](xattn_kernels_guide.md): Triton kernels 实现
- [`docs/sparse_policy_architecture.md`](sparse_policy_architecture.md): SparsePolicy 架构
- [`docs/sparse_policy_implementation_guide.md`](sparse_policy_implementation_guide.md): 实现指南
