# Sparse Attention Computation Flow Guide

This document explains the computation flow for block sparse attention methods used in long-context LLM inference. The implementations are from the x-attention project.

## Overview

All sparse attention methods follow a common pattern:

```
1. Estimate important blocks (low-cost estimation)
2. Create block mask (True = compute, False = skip)
3. Execute block sparse attention kernel (only compute selected blocks)
```

The key difference between methods is **how they estimate which blocks are important**.

---

## Common Concepts

### Block Tiling

All methods divide Q, K, V into blocks:
- **Block Size**: Typically 64 or 128 tokens per block
- **Num Q Blocks**: `ceil(seq_len / block_size)`
- **Num K Blocks**: `ceil(seq_len / block_size)`
- **Block Mask**: Boolean tensor `[batch, heads, q_blocks, k_blocks]`

### Causal Masking

For autoregressive models, block `(i, j)` is only valid if `j <= i` (causal constraint).

### Block Sparse Attention Kernel

All methods ultimately call `block_sparse_attn_func` from MIT-HAN-LAB:
```python
from block_sparse_attn import block_sparse_attn_func

output = block_sparse_attn_func(
    query_states,      # [batch, seq, heads, head_dim]
    key_states,        # [batch, seq, heads, head_dim]
    value_states,      # [batch, seq, heads, head_dim]
    block_mask,        # [batch, heads, q_blocks, k_blocks] bool
    block_size=64,     # tokens per block
    causal=True,
)
```

---

## Method 1: XAttention (xattn_estimate)

**Source**: `compass/src/Xattention.py`

**详细文档**: [`docs/xattention_algorithm_guide.md`](xattention_algorithm_guide.md)

### Core Idea

Use **stride interleaved reshape (inverse mode)** to efficiently estimate block-level attention importance, then use **BSA (Block Sparse Attention)** library for sparse computation.

### Algorithm

```python
def xattn_estimate(query, key, block_size=128, stride=8):
    """
    Estimate block importance using stride-interleaved attention.

    1. K reshape (正向交错): concat([K[:,:,k::stride,:] for k in range(stride)])
       Q reshape (反向交错): concat([Q[:,:,(stride-1-q)::stride,:] for q])
       结果: 序列长度 seq_len -> seq_len/stride, head_dim -> head_dim*stride

    2. Triton kernel (flat_group_gemm_fuse_reshape):
       融合 reshape + GEMM，计算 Q_reshaped @ K_reshaped^T

    3. Triton kernel (softmax_fuse_block_sum):
       在线 softmax + 按 block_size/stride 分组求和
       输出: attn_sum [batch, heads, q_blocks, k_blocks]

    4. find_blocks_chunked:
       按 attn_sum 降序排序，累积到 threshold 的块标记为 True
       对角块和 sink 块始终保留
    """
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 128 | Tokens per block (BSA 要求固定 128) |
| `stride` | 8 | Q/K 交错采样步长，越大估计越快但越粗糙 |
| `threshold` | 0.9 | 累积注意力阈值，选择累积权重达到此比例的块 |
| `chunk_size` | 16384 | 估计时的分块大小 |

### Computation Flow

```
query [B, H, S, D]
    |
    v
Stride interleaved reshape (Triton fused)
    |
    v
flat_group_gemm_fuse_reshape: Q_r @ K_r^T
    |
    v
softmax_fuse_block_sum: 在线 softmax + 块求和
    |
    v
attn_sum [B, H, q_blocks, k_blocks]
    |
    v
find_blocks_chunked: 累积阈值选择
    |
    v
simple_mask [B, H, q_blocks, k_blocks] (bool)
    |
    v
block_sparse_attn_func(q, k, v, simple_mask)  ← BSA 库
    |
    v
output [B, H, S, D]
```

### Dependencies

```python
from block_sparse_attn import block_sparse_attn_func  # MIT-HAN-LAB BSA 库
import triton  # Triton kernels for estimation
```

### Usage

```python
from compass.src.Xattention import Xattention_prefill

output = Xattention_prefill(
    query_states, key_states, value_states,
    threshold=0.9,
    stride=8,
    block_size=128,
    use_triton=True,
)

---

## Method 2: FlexPrefill

**Source**: `xattn/src/Flexprefill.py`

### Core Idea

Use **last-q attention pattern** to detect vertical (column-wise important) and slash (diagonal-like) patterns. Adaptively adjust budget based on **JS divergence** between estimated and uniform distribution.

### Algorithm

```python
def Flexprefill_prefill(query, key, value, gamma=0.9, tau=0.1):
    """
    1. Compute attention using only last 64 queries (last_q_attn)
       This reveals which K positions are globally important

    2. Detect vertical pattern: columns with high attention across all last-q
       These are "sink tokens" that all queries attend to

    3. Detect slash pattern: diagonal bands that capture local attention

    4. Compute JS divergence between estimated pattern and uniform
       - High divergence = sparse pattern detected, use fewer blocks
       - Low divergence = dense pattern needed, use more blocks

    5. Adjust budget per head based on divergence

    6. Select top-scoring blocks up to budget
    """
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.9 | Base coverage ratio (fraction of blocks to keep) |
| `tau` | 0.1 | JS divergence threshold for adaptive budget |
| `min_budget` | 0.5 | Minimum coverage even for sparse patterns |
| `block_size` | 64 | Tokens per block |

### Patterns Detected

1. **Vertical Pattern**: Columns where many queries attend heavily
   - Detected by summing attention across query dimension
   - Captures "attention sinks" (e.g., BOS token, punctuation)

2. **Slash Pattern**: Diagonal bands
   - Captures local context attention
   - Width determined by `slash_size` parameter

### Computation Flow

```
query [B, S, H, D]
    |
    v
Take last 64 queries -> last_q [B, 64, H, D]
    |
    v
Compute last_q attention to all K -> attn [B, H, 64, S]
    |
    v
Analyze pattern:
  - Vertical: sum over query dim, find high columns
  - Slash: diagonal bands
    |
    v
Compute JS divergence per head
    |
    v
Adaptive budget = gamma * (1 - divergence/tau)
    |
    v
Select blocks up to budget -> block_mask
    |
    v
block_sparse_attn_func(q, k, v, block_mask)
    |
    v
output [B, S, H, D]
```

### Triton Kernels

FlexPrefill uses custom Triton kernels for efficiency:
- `flex_prefill_attention_kernel`: Block-wise attention with pattern masking
- `flex_vertical_slash_kernel`: Combined vertical + slash pattern attention

---

## Method 3: MInference

**Source**: `xattn/src/Minference.py`

### Core Idea

Simple and direct: use **vertical_slash_sparse_attention** kernel with pre-computed vertical and slash indices.

### Algorithm

```python
def Minference_prefill(query, key, value, vertical_topk=100):
    """
    1. Compute attention from last 64 queries to all K

    2. For each head, identify:
       - vertical_indices: top-k columns with highest total attention
       - slash_indices: diagonal band positions

    3. Call vertical_slash_sparse_attention kernel
       - Computes attention only at selected positions
       - Returns output with zeros elsewhere
    """
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vertical_topk` | 100 | Number of important columns to select |
| `slash_n` | 64 | Size of diagonal band |

### Computation Flow

```
query [B, S, H, D]
    |
    v
Take last 64 queries
    |
    v
Compute attention scores to all K
    |
    v
Sum across queries -> column importance [H, S]
    |
    v
Select top-k columns per head -> vertical_indices [H, topk]
    |
    v
Generate slash indices (diagonal positions) -> slash_indices [H, slash_n]
    |
    v
vertical_slash_sparse_attention(q, k, v, vertical_indices, slash_indices)
    |
    v
output [B, S, H, D]
```

### CUDA Kernel

MInference uses a specialized CUDA kernel:
```python
from minference.ops.block_sparse_attn import vertical_slash_sparse_attention

output = vertical_slash_sparse_attention(
    query, key, value,
    vertical_indices,  # [heads, topk]
    slash_indices,     # [heads, slash_n]
)
```

---

## Method 4: AvgPool

**Source**: `xattn/src/AvgPool.py`

### Core Idea

Pool Q and K within each block using average pooling, compute block-level softmax attention, then select blocks using **top-k** or **top-p (nucleus sampling)**.

### Algorithm

```python
def AvgPool_prefill(query, key, value, block_size=128, top_k=64, top_p=None):
    """
    1. Divide Q into blocks, apply average pooling -> block_q [B, q_blocks, H, D]

    2. Divide K into blocks, apply average pooling -> block_k [B, k_blocks, H, D]

    3. Compute block attention: softmax(block_q @ block_k.T / sqrt(d))
       Result: [B, H, q_blocks, k_blocks]

    4. Select blocks:
       - top_k: Select k highest scoring blocks per row
       - top_p: Sort descending, accumulate until sum > p, select all accumulated

    5. Create block_mask from selection

    6. Execute block_sparse_attn_func
    """
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 128 | Tokens per block |
| `chunk_size` | 16384 | Processing chunk size |
| `top_k` | 64 | Fixed number of blocks per row |
| `top_p` | None | Cumulative probability threshold (0.0-1.0) |
| `pool_method` | "avg" | Pooling method (avg, max) |

### Top-K vs Top-P Selection

**Top-K**: Select exactly k blocks with highest scores per row.
```
scores = [0.4, 0.3, 0.2, 0.1], k=2
selected = [0.4, 0.3]  # indices 0, 1
```

**Top-P (Nucleus)**: Sort descending, accumulate until exceeding threshold.
```
scores = [0.4, 0.3, 0.2, 0.1], p=0.8
cumsum = [0.4, 0.7, 0.9, 1.0]
selected = [0.4, 0.3, 0.2]  # first 3 (cumsum exceeds 0.8 at index 2)
```

### Computation Flow

```
query [B, S, H, D]
    |
    v
Reshape to blocks [B, num_blocks, block_size, H, D]
    |
    v
Average pool -> block_q [B, num_blocks, H, D]
    |
    v
Same for K -> block_k [B, num_blocks, H, D]
    |
    v
Compute block scores [B, H, q_blocks, k_blocks]
    |
    v
Apply causal mask
    |
    v
top_k or top_p selection -> block_mask
    |
    v
block_sparse_attn_func(q, k, v, block_mask)
    |
    v
output [B, S, H, D]
```

---

## Comparison

| Method | Estimation Cost | Adaptivity | Typical Sparsity |
|--------|-----------------|------------|------------------|
| XAttention | Medium (strided attn) | Threshold-based | 60-80% |
| FlexPrefill | Medium (last-q attn) | JS divergence | 50-70% |
| MInference | Low (last-q attn) | Fixed vertical+slash | 70-90% |
| AvgPool | Medium (pooled attn) | top-k/top-p | 50-80% |

### When to Use Each

- **XAttention**: General purpose, good balance of accuracy and speed
- **FlexPrefill**: When pattern varies significantly across heads
- **MInference**: When vertical (sink) and local (slash) patterns dominate
- **AvgPool**: When you want simple, interpretable block selection

---

## Integration with HuggingFace Models

All methods integrate via `FastPrefillConfig` in `xattn/src/load_llama.py`:

```python
from xattn.src.load_llama import FastPrefillConfig, load_model_and_apply_fastprefill

config = FastPrefillConfig(
    metric="xattn",      # "xattn", "flexprefill", "minference", "avgpool"
    threshold=0.9,       # for xattn
    stride=16,           # for xattn
    top_k=64,            # for avgpool
    top_p=None,          # for avgpool nucleus sampling
)

model, tokenizer = load_model_and_apply_fastprefill(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    fastprefillconfig=config,
)
```

The `forward` method of attention layers is monkey-patched to use the selected sparse attention method during prefill.

---

## Key Files

| File | Purpose |
|------|---------|
| `xattn/src/Xattention.py` | XAttention implementation |
| `xattn/src/Flexprefill.py` | FlexPrefill implementation |
| `xattn/src/Minference.py` | MInference implementation |
| `xattn/src/AvgPool.py` | AvgPool implementation |
| `xattn/src/load_llama.py` | Model loading and method dispatch |
| `xattn/src/Compass.py` | Another sparse method (gradient-based) |

---

## Dependencies

Required libraries:
- `torch`: PyTorch
- `triton`: For FlexPrefill Triton kernels
- `flash_attn`: Flash Attention for baseline
- `block_sparse_attn`: MIT-HAN-LAB block sparse kernel
- `minference`: For MInference vertical_slash kernel

Docker image `tzj/xattn:v0.5` has all dependencies pre-installed.

---

## Quest Sparse Policy

**Files**: `nanovllm/kvcache/sparse/quest.py`, `nanovllm/kvcache/sparse/policy.py`

### Core Idea

Quest policy selects Top-K blocks based on query-key similarity bounds using min/max key metadata. This enables efficient block selection for CPU offload scenarios.

### Scoring Mechanism

```python
# Compute scores using key metadata bounds
score_min = torch.einsum('hd,bhd->bh', q, key_min)  # [num_blocks, kv_heads]
score_max = torch.einsum('hd,bhd->bh', q, key_max)  # [num_blocks, kv_heads]
scores = torch.maximum(score_min, score_max).mean(dim=-1)  # [num_blocks] ← averaged!
```

### Critical Limitation - No Per-Head Scheduling

The `.mean(dim=-1)` averages scores across all heads, making a **unified** block selection for all heads:

```
Block A: head0 needs (+4), head1 doesn't (-4) → avg = 0 → NOT selected
Block B: head0 doesn't (-4), head1 needs (+4) → avg = 0 → NOT selected
Block C: both heads moderately need (+2, +2) → avg = +2 → selected
```

### Why Per-Head Scheduling is Infeasible

1. **Memory Layout**: GPU cache stores all heads together `[block_size, kv_heads, head_dim]`

2. **FlashAttention**: Requires complete heads - partial heads cause dimension mismatch

3. **Block Granularity**: If any head needs a block, the entire block (all heads) must be loaded

### Policy Types

| Policy | supports_prefill | supports_decode | Description |
|--------|------------------|-----------------|-------------|
| `FullAttentionPolicy` | True | True | Loads all blocks (no sparsity) |
| `QuestPolicy` | False | True | Decode-only Top-K selection |

### Usage Example

```python
from nanovllm.kvcache.sparse.policy import QuestPolicy

# Create Quest policy for decode-only sparse attention
policy = QuestPolicy(topk=8, threshold=4.0)

# Select blocks based on query and key metadata
selected_blocks = policy.select_blocks(
    query,           # [num_tokens, num_heads, head_dim]
    key_min,         # [num_blocks, num_heads, head_dim]
    key_max,         # [num_blocks, num_heads, head_dim]
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topk` | 8 | Number of blocks to select |
| `threshold` | 4.0 | Minimum score threshold for selection |

### Integration with CPU Offload

The Quest policy is used in conjunction with CPU offload to reduce the number of blocks transferred from CPU to GPU during decode:

1. During prefill, all blocks are loaded (full attention)
2. During decode, Quest selects only top-K important blocks
3. Only selected blocks are transferred from CPU to GPU
4. This reduces memory bandwidth requirements for long sequences
