# Block Sparse Attention Interface

Source: [MIT-HAN-LAB/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)

This document records the BSA (Block Sparse Attention) interface used by XAttention for sparse attention computation.

## Installation

BSA is installed in the `minference` conda environment:
```
/home/zijie/anaconda3/envs/minference/lib/python3.10/site-packages/block_sparse_attn/
```

To use in other environments, add to PYTHONPATH:
```bash
PYTHONPATH=/home/zijie/anaconda3/envs/minference/lib/python3.10/site-packages:$PYTHONPATH python script.py
```

## Interface Code

```python
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import block_sparse_attn_cuda
import torch
import torch.nn as nn


def convert_blockmask(blockmask, causal):
    """Convert from the 0-1 format to the format used by the CUDA code.
    0 means the block is skipped.
    nonzero means the block is not skipped.
    Argument:
        blockmask: (row, col): a 0-1 tensor
    Return:
        blockmask_converted: (col, row), dtype torch.int32: for each column, it contains the row
            indices of the nonzero blocks, padded with -1 to reach length @row.
            The indices are multiplied by 4, with the smallest bit used to encode whether
            it is the first nonzero in its row, and the 2nd smallest bit to encode whether it is
            the last nonzero in its row..
    """
    assert not causal
    nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=0, stable=True, descending=True)
    nonzero_unsorted_rowidx = nonzero_sorted_rowidx.argsort(dim=0)
    last_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True).indices[:, -1]
    last_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), last_nonzero_col_per_row
    ]
    first_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True, descending=True).indices[:, 0]
    first_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), first_nonzero_col_per_row
    ]
    nonzero_idx = nonzero_sorted_rowidx * 4
    nonzero_idx[last_nonzero_col_per_row_after_sort, last_nonzero_col_per_row] += 2
    nonzero_idx[first_nonzero_col_per_row_after_sort, first_nonzero_col_per_row] += 1
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.T.contiguous().to(dtype=torch.int32)


def convert_blockmask_row_reverse(blockmask, causal=False):
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-1, stable=True, descending=False)

    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])

    return nonzero_idx.contiguous().to(dtype=torch.int32)


def convert_blockmask_col_reverse(blockmask, causal=False):
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-2, stable=True, descending=False)

    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-2])
    nonzero_idx = torch.transpose(nonzero_idx, -1, -2)

    return nonzero_idx.contiguous().to(dtype=torch.int32)


def replace_ones_with_count(tensor):
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def _block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    row_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right
):
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        row_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_softmax,
        window_size_left,
        window_size_right,
        None
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def block_sparse_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
    """
    Main entry point for block sparse attention.

    Args:
        q: Query tensor [total_q, num_heads, head_dim]
        k: Key tensor [total_k, num_heads, head_dim]
        v: Value tensor [total_k, num_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for Q [batch+1]
        cu_seqlens_k: Cumulative sequence lengths for K [batch+1]
        head_mask_type: Per-head mask type [num_heads], 1 for block sparse
        streaming_info: Optional streaming attention info
        base_blockmask: Block mask [batch, num_heads, q_blocks, k_blocks]
        max_seqlen_q_: Maximum Q sequence length
        max_seqlen_k_: Maximum K sequence length
        p_dropout: Dropout probability (0.0 for eval)
        deterministic: Whether to use deterministic algorithms
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
        exact_streaming: Whether to use exact streaming attention
        return_attn_probs: Whether to return attention probabilities

    Returns:
        Attention output [total_q, num_heads, head_dim]
    """
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num

    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,  # m_block_dim, n_block_dim (fixed at 128)
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_attn_probs,
                -1, -1,  # window_size_left, window_size_right
                deterministic
                )
```

## Usage Example (from COMPASS)

```python
from block_sparse_attn import block_sparse_attn_func

# After xattn_estimate returns sparse mask
attn_sums, approx_simple_mask = xattn_estimate(query_states, key_states, ...)

# Reshape for BSA (requires [seq_len, num_heads, head_dim] format)
query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)

# Cumulative sequence lengths
q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=device)

# Head mask type (1 for all heads using block sparse)
head_mask_type = torch.tensor([1] * num_heads, device=device, dtype=torch.int32)

# Call BSA
attn_output = block_sparse_attn_func(
    query_states,
    key_states,
    value_states,
    q_cu_seq_lens,
    k_cu_seq_lens,
    head_mask_type,
    None,  # streaming_info
    approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
    q_len,
    k_len,
    p_dropout=0.0,
    deterministic=True,
    is_causal=True,
)

# Reshape back to [batch, num_heads, seq_len, head_dim]
attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
```

## Key Constraints

- **Block size**: Fixed at 128 tokens (hardcoded in BSA)
- **Batch size**: Only batch_size=1 supported for block sparse mode
- **Mask format**: `[batch, num_heads, q_blocks, k_blocks]` boolean tensor
- **Input format**: `[total_seq_len, num_heads, head_dim]` (not batched)
