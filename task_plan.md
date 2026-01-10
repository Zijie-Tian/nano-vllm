# Task Plan: Integrate Three Sparse Attention Strategies

## Executive Summary

This plan addresses integrating MInference, FlexPrefill, and XAttention sparse attention strategies into nanovllm. The analysis reveals fundamental architectural issues with the current MInference integration that must be resolved before adding the other strategies.

---

## Current State Analysis

### x-attention Repository Structure

| Strategy | File | Estimation Method | Execution Kernel |
|----------|------|-------------------|------------------|
| **MInference** | `Minference.py` | Last-64 Q, vertical+slash indices | `vertical_slash_sparse_attention` (minference lib) |
| **XAttention** | `Xattention.py` | Strided Q/K sampling, block scores | `block_sparse_attn_func` (MIT-HAN-LAB) |
| **FlexPrefill** | `Flexprefill.py` | Last-block Q, JS divergence budget | `triton_block_wise_attention` (custom Triton) |

### Key Insight: Two Different Kernel Interfaces

**Interface A: Index-Based (minference library)**
```python
# MInference uses vertical+slash indices
vertical_indices = [heads, vertical_size]  # Important K column positions
slash_indices = [heads, slash_size]        # Diagonal offsets
output = vertical_slash_sparse_attention(q, k, v, vertical_indices, slash_indices)
```

**Interface B: Block Mask-Based (block_sparse_attn)**
```python
# XAttention/FlexPrefill use boolean block mask
block_mask = torch.bool[batch, heads, q_blocks, k_blocks]  # True = compute
output = block_sparse_attn_func(q, k, v, block_mask, ...)
```

### Problems with Current nanovllm MInference Integration

1. **Locked to Minference Kernel**: Uses `_triton_mixed_sparse_attention` from minference library
2. **Cannot Share with XAttention/FlexPrefill**: Different kernel interface
3. **Monolithic API**: `sparse_prefill_attention()` mixes estimation and execution
4. **No Block Mask Output**: Cannot reuse pattern estimation for different kernels

---

## Architectural Decision

### Option A: Maintain Two Execution Paths
- MInference → `vertical_slash_sparse_attention`
- XAttention/FlexPrefill → `block_sparse_attn_func`

**Pros**: Minimal changes to existing MInference code
**Cons**: Code duplication, no unified interface, harder to maintain

### Option B: Unify to Block Mask Interface (RECOMMENDED)
- All strategies → Boolean block mask → `block_sparse_attn_func`
- MInference's vertical+slash pattern can be converted to block mask

**Pros**: Single execution path, unified interface, easier testing, better extensibility
**Cons**: Need to rewrite MInference execution, potential minor accuracy difference

**Decision**: Option B - Unify to block mask interface

---

## Target Architecture

### Unified Sparse Policy Interface

```python
class SparsePrefillPolicy(ABC):
    """Base class for prefill sparse attention policies."""

    # Core estimation method - MUST implement
    @abstractmethod
    def estimate_block_mask(
        self,
        q: torch.Tensor,           # [seq_len, num_heads, head_dim]
        k: torch.Tensor,           # [seq_len, num_kv_heads, head_dim]
        block_size: int,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Estimate which blocks are important.

        Returns:
            block_mask: [batch, heads, q_blocks, k_blocks] boolean tensor
                       True = compute attention, False = skip
        """
        pass

    # Optional: custom sparse attention (default uses block_sparse_attn_func)
    def sparse_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """
        Full sparse prefill: estimate + execute.
        Default implementation uses block_sparse_attn_func.
        """
        block_mask = self.estimate_block_mask(q, k, self.block_size, layer_id)
        return block_sparse_attention(q, k, v, block_mask, self.block_size)
```

### Unified Execution Kernel

```python
def block_sparse_attention(
    q: torch.Tensor,           # [seq_len, num_heads, head_dim]
    k: torch.Tensor,           # [seq_len, num_kv_heads, head_dim]
    v: torch.Tensor,           # [seq_len, num_kv_heads, head_dim]
    block_mask: torch.Tensor,  # [batch, heads, q_blocks, k_blocks]
    block_size: int = 128,
) -> torch.Tensor:
    """
    Execute block sparse attention using MIT-HAN-LAB kernel.
    Wrapper around block_sparse_attn_func with nanovllm tensor conventions.
    """
    from block_sparse_attn import block_sparse_attn_func
    # ... reshape and call kernel
```

---

## Strategy-Specific Block Mask Generation

### 1. MInference Block Mask

**Algorithm**: Convert vertical+slash indices to block mask

```python
def minference_estimate_block_mask(q, k, block_size, layer_id):
    # Step 1: Compute vertical and slash patterns (existing code)
    vertical_indices, slash_indices = estimate_pattern(q, k, layer_id)

    # Step 2: Convert to block mask
    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = torch.zeros(1, num_heads, num_blocks, num_blocks, dtype=bool)

    # Vertical: column positions → block columns for all rows
    vertical_block_cols = vertical_indices // block_size  # [heads, top_v]
    for h in range(num_heads):
        block_mask[0, h, :, vertical_block_cols[h]] = True

    # Slash: diagonal offsets → diagonal bands in block space
    for h in range(num_heads):
        for diag_offset in slash_indices[h]:
            # Set diagonal blocks with this offset
            for i in range(num_blocks):
                j = i - diag_offset // block_size
                if 0 <= j < num_blocks:
                    block_mask[0, h, i, j] = True

    # Ensure causal + diagonal
    block_mask = apply_causal_mask(block_mask)
    return block_mask
```

### 2. XAttention Block Mask

**Algorithm**: Strided sampling for fast block scoring (from x-attention)

```python
def xattn_estimate_block_mask(q, k, block_size, stride, threshold, layer_id):
    """
    Direct port of xattn_estimate from x-attention repo.
    """
    # Step 1: Strided reshape for coarse Q/K
    # q_strided: take every `stride`-th token, reshape to represent blocks

    # Step 2: Compute block-level attention scores
    block_scores = softmax(block_q @ block_k.T / sqrt(d))  # [heads, q_blocks, k_blocks]

    # Step 3: Threshold selection
    # Sort scores, accumulate until reaching threshold (cumulative sum approach)
    block_mask = find_blocks_chunked(block_scores, threshold)

    return block_mask
```

### 3. FlexPrefill Block Mask

**Algorithm**: Last-block attention + JS divergence adaptive budget

```python
def flexprefill_estimate_block_mask(q, k, block_size, gamma, tau, layer_id):
    """
    Port of get_active_blocks from x-attention repo.
    """
    # Step 1: Compute last-block attention
    last_q = q[-block_size:]
    qk = softmax(last_q @ k.T / sqrt(d))  # [block_size, seq_len]

    # Step 2: Extract vertical and slash patterns
    vertical = qk.sum(dim=0)  # Column importance
    slash = sum_all_diagonal_matrix(qk)  # Diagonal importance

    # Step 3: Compute JS divergence for adaptive budget per head
    avg_k_pooled = avg_pool(k, block_size)
    avg_qk = softmax(last_q.mean(0) @ avg_k_pooled.T)
    kl_div = js_divergence(avg_qk, vertical_pooled)

    # Step 4: Per-head budget adjustment
    is_sparse_head = kl_div > tau
    budget = gamma if is_sparse_head else 1.0

    # Step 5: Select top blocks up to budget
    # Combine vertical_topk and slash_topk
    block_idx = transform_vertical_slash_idx(vertical_topk, slash_topk, num_blocks)

    return block_mask_from_indices(block_idx, num_blocks)
```

---

## Implementation Plan

### Phase 1: Add Block Sparse Attention Wrapper

**Files to create/modify:**
- `nanovllm/kvcache/sparse/block_sparse_kernel.py` (NEW)

**Tasks:**
1. Create wrapper around `block_sparse_attn_func` with nanovllm conventions
2. Handle GQA (expand K/V heads to match Q heads)
3. Handle tensor format conversion: `[seq, heads, dim]` ↔ `[seq, heads, dim]`
4. Test with simple identity block mask (full attention)

### Phase 2: Refactor MInference to Block Mask

**Files to modify:**
- `nanovllm/kvcache/sparse/minference.py`

**Tasks:**
1. Add `estimate_block_mask()` method that returns `[1, heads, q_blocks, k_blocks]`
2. Convert vertical+slash indices to block mask format
3. Replace `sparse_prefill_attention()` to use new unified kernel
4. Test correctness with needle-in-haystack

### Phase 3: Add XAttention Policy

**Files to create:**
- `nanovllm/kvcache/sparse/xattention.py` (NEW)

**Tasks:**
1. Port `xattn_estimate` from x-attention (strided sampling logic)
2. Port Triton kernels if beneficial (`flat_group_gemm`, `softmax_fuse_block_sum`)
3. Implement `estimate_block_mask()` following unified interface
4. Test and benchmark vs MInference

### Phase 4: Add FlexPrefill Policy

**Files to create:**
- `nanovllm/kvcache/sparse/flexprefill.py` (NEW)

**Tasks:**
1. Port `get_active_blocks` from x-attention
2. Port helper functions: `sum_all_diagonal_matrix`, `js_divergence`, `transform_vertical_slash_idx`
3. Port Triton kernels: `triton_bnhd_pool`, `triton_bhn_sumpool`
4. Implement `estimate_block_mask()` following unified interface
5. Test and benchmark

### Phase 5: Configuration and Integration

**Files to modify:**
- `nanovllm/config.py`
- `nanovllm/kvcache/sparse/__init__.py`
- `nanovllm/engine/model_runner.py`

**Tasks:**
1. Add `SparsePolicyType.XATTENTION`, `SparsePolicyType.FLEXPREFILL`
2. Add policy-specific config parameters
3. Update policy factory to support all three
4. Update documentation

### Phase 6: Testing and Benchmarking

**Tasks:**
1. Correctness tests for all three policies
2. Performance benchmarks (prefill throughput vs full attention)
3. Memory usage comparison
4. Accuracy comparison (output quality)

---

## Configuration Parameters

### MInference
```python
sparse_policy = SparsePolicyType.MINFERENCE
minference_adaptive_budget = 0.3      # Budget as fraction of seq_len
minference_vertical_size = 1000       # Fixed vertical (if budget=None)
minference_slash_size = 6096          # Fixed slash (if budget=None)
```

### XAttention
```python
sparse_policy = SparsePolicyType.XATTENTION
xattn_stride = 16                     # Stride for coarse Q/K sampling
xattn_threshold = 0.9                 # Cumulative score threshold
xattn_block_size = 128                # Block size (must match kernel)
```

### FlexPrefill
```python
sparse_policy = SparsePolicyType.FLEXPREFILL
flex_gamma = 0.9                      # Base coverage ratio
flex_tau = 0.1                        # JS divergence threshold
flex_min_budget = 128                 # Minimum tokens per head
flex_max_budget = None                # Maximum tokens per head
```

---

## Dependencies

### Required Libraries
```
block_sparse_attn        # MIT-HAN-LAB block sparse kernel
triton                   # For XAttention/FlexPrefill Triton kernels
```

### Optional (for comparison/fallback)
```
minference               # For MInference vertical_slash kernel (fallback)
```

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Block mask conversion loses MInference accuracy | Medium | Low | Test needle accuracy, compare outputs |
| block_sparse_attn kernel compatibility | High | Medium | Test on target hardware, fallback to minference |
| Triton kernel portability | Medium | Medium | Use non-Triton fallback paths |
| Memory overhead of block mask | Low | Low | block_size=128 → 1KB per head for 128K seq |

---

## Success Criteria

1. **Correctness**: All three policies pass needle-in-haystack test at 32K+ length
2. **Performance**: Sparse prefill faster than full attention (>1.5x speedup at 64K)
3. **Unified Interface**: All policies implement `estimate_block_mask()` with same signature
4. **Configurability**: All policy parameters exposed via LLM config

---

## Timeline Estimate

| Phase | Complexity | Status |
|-------|------------|--------|
| Phase 1: Block Sparse Wrapper | Low | Pending |
| Phase 2: Refactor MInference | Medium | Pending |
| Phase 3: XAttention | Medium | Pending |
| Phase 4: FlexPrefill | High | Pending |
| Phase 5: Integration | Low | Pending |
| Phase 6: Testing | Medium | Pending |

---

## Appendix: Key Functions to Port from x-attention

### From `Xattention.py`
- `xattn_estimate()` - Main strided estimation
- `find_blocks_chunked()` (from `utils.py`) - Threshold selection
- `flat_group_gemm_fuse_reshape()` - Triton kernel (optional)
- `softmax_fuse_block_sum()` - Triton kernel (optional)

### From `Flexprefill.py`
- `get_active_blocks()` - Main estimation
- `sum_all_diagonal_matrix()` - Diagonal pattern extraction
- `square_root_js_divergence()` - Adaptive budget
- `transform_vertical_slash_idx()` - Index to block conversion
- `triton_bnhd_pool()` - Block pooling kernel
- `triton_block_wise_attention()` - Custom sparse attention

### From `block_sparse_attn_interface.py`
- `block_sparse_attn_func()` - Main kernel interface
- `convert_blockmask_row_reverse()` - Block mask format conversion

---

## Appendix: MInference 可使用 block_sparse_attn 的可行性分析

### 核心结论

**MInference 的 vertical+slash 稀疏模式完全可以映射到 block mask，从而使用 `block_sparse_attn_func` kernel。**

### Vertical Pattern → Block Mask

MInference 选择的 vertical 位置（重要的 K 列）可以直接映射到包含这些位置的 K blocks：

```python
# Token-level: 选择位置 [0, 5, 128, 130, 256, ...]
vertical_indices = [0, 5, 128, 130, 256]

# Block-level: 转换为包含这些位置的 blocks (block_size=128)
vertical_blocks = set(idx // block_size for idx in vertical_indices)
# → {0, 1, 2}

# 对所有 Q blocks，都 attend 这些 K blocks
block_mask[:, :, :, list(vertical_blocks)] = True
```

### Slash Pattern → Block Mask

MInference 的 slash pattern（对角线带状区域）天然是 block-aligned 的：

```python
# 对角线区域：query i attend [max(0, i-w), i]
# 在 block 空间中就是对角线 + 下方几个 blocks

for q_block in range(num_q_blocks):
    slash_width_blocks = slash_size // block_size
    for k_block in range(max(0, q_block - slash_width_blocks), q_block + 1):
        block_mask[:, :, q_block, k_block] = True
```

### 图示对比

```
Token-level MInference:              Block-level mask:
┌─────────────────────────┐          ┌───┬───┬───┬───┐
│■ ■         ■            │          │ ■ │ ■ │   │   │  ← vertical blocks
│■ ■ ■       ■            │          ├───┼───┼───┼───┤
│■ ■ ■ ■     ■            │    →     │ ■ │ ■ │ ■ │   │  ← diagonal + vertical
│■ ■   ■ ■   ■            │          ├───┼───┼───┼───┤
│■ ■     ■ ■ ■            │          │ ■ │ ■ │ ■ │ ■ │
└─────────────────────────┘          └───┴───┴───┴───┘
  ↑ vertical    ↑ slash               Block granularity (128 tokens/block)
```

### 粒度差异分析

| Kernel | 粒度 | 精度 | 计算量 |
|--------|------|------|--------|
| `vertical_slash_sparse_attention` (minference) | Token-level | 精确 | 最少 |
| `block_sparse_attn_func` (MIT-HAN-LAB) | Block-level (128) | 近似 | 稍多 |

**Block-level 会多计算的情况**：
- 例如 vertical 选了位置 5，但整个 block 0 (位置 0-127) 都会被计算

**但实际影响很小**：
1. Vertical 位置通常集中在开头 (sink tokens) → 只影响前几个 blocks
2. Slash pattern 本身就是连续区域，block 对齐损失很小
3. 实际稀疏比例差异通常 < 5% (如从 80% 稀疏变成 75% 稀疏)

### 未来实现方向

如果选择将 MInference 迁移到 `block_sparse_attn_func`：

```python
class MInferencePolicy(SparsePrefillPolicy):
    def estimate_block_mask(self, q, k, block_size, layer_id):
        # Step 1: 使用现有 estimate_pattern() 获取 token-level indices
        vertical_indices, slash_indices = self.estimate_pattern(q, k, layer_id)

        # Step 2: 转换为 block mask
        num_blocks = (seq_len + block_size - 1) // block_size
        block_mask = torch.zeros(1, num_heads, num_blocks, num_blocks, dtype=torch.bool)

        # Vertical: token positions → block columns
        for h in range(num_heads):
            v_blocks = torch.unique(vertical_indices[h] // block_size)
            block_mask[0, h, :, v_blocks] = True

        # Slash: diagonal offsets → diagonal block bands
        for h in range(num_heads):
            for diag in slash_indices[h]:
                diag_block = diag // block_size
                for i in range(num_blocks):
                    j = i - diag_block
                    if 0 <= j < num_blocks:
                        block_mask[0, h, i, j] = True

        # Apply causal mask
        causal = torch.tril(torch.ones(num_blocks, num_blocks, dtype=torch.bool))
        block_mask &= causal

        return block_mask
```

### 优势

1. **统一 kernel**：三种策略使用同一个 `block_sparse_attn_func`
2. **减少依赖**：不再需要 minference 库
3. **更好的可维护性**：只需维护一套 kernel 调用代码
4. **硬件兼容性**：`block_sparse_attn` 可能有更好的硬件支持

### 潜在风险

1. **精度损失**：Block 粒度可能导致轻微精度下降（需要验证）
2. **性能变化**：计算量稍多，但 kernel 效率可能更高（需要 benchmark）

---

## References

- x-attention repo: `/home/zijie/Code/x-attention`
- MIT-HAN-LAB Block-Sparse-Attention: `/home/zijie/Code/x-attention/Block-Sparse-Attention`
- MInference paper: https://arxiv.org/abs/2407.02490
- FlexPrefill paper: (from x-attention implementation)
- XAttention paper: (from x-attention implementation)
