"""
AvgPool: Block-level Sparse Attention Estimation using Pooled Q/K

This module implements sparse attention estimation by pooling query and key
tensors at block level, computing block attention scores, and selecting
top-k important blocks per query block row.

Estimation Strategy:
- Pool Q/K tensors within each block (avg or max pooling)
- Compute block-level attention: softmax(Q_pool @ K_pool.T / sqrt(d))
- For each query block row, select top-k key blocks with highest attention

Structure:
- AvgPool_estimate(): Estimates important blocks using pooled attention
- AvgPool_prefill(): Uses the estimate to perform block sparse attention
"""

import torch
import torch.nn.functional as F
import math
from block_sparse_attn import block_sparse_attn_func


def AvgPool_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int = 128,
    chunk_size: int = 16384,
    top_k: int = None,
    top_p: float = None,
    causal: bool = True,
    pool_method: str = "avg",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate block importance using pooled Q/K attention.

    For each query block, pools the queries and keys within each block,
    computes attention scores between pooled representations, and selects
    top-k or top-p key blocks per query block row.

    Args:
        query_states: Query tensor [batch, num_heads, q_len, head_dim]
        key_states: Key tensor [batch, num_heads, k_len, head_dim]
        block_size: Size of each attention block (default: 128)
        chunk_size: Padding alignment for block calculation (default: 16384)
        top_k: Number of top blocks to select per query block row (mutually exclusive with top_p)
        top_p: Cumulative probability threshold for nucleus sampling (mutually exclusive with top_k)
        causal: Whether to apply causal masking
        pool_method: "avg" for mean pooling, "max" for max pooling

    Returns:
        block_scores: [batch, num_heads, q_blocks, k_blocks] - softmax attention per block
        block_mask: [batch, num_heads, q_blocks, k_blocks] - True = keep (selected blocks)
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    # Calculate number of blocks with padding alignment
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len

    k_block_num = (k_len + k_num_to_pad) // block_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    # Actual blocks (without chunk padding, just block alignment)
    actual_k_blocks = math.ceil(k_len / block_size)
    actual_q_blocks = math.ceil(q_len / block_size)

    scale = 1.0 / math.sqrt(head_dim)

    # Pad Q and K to align with block boundaries
    q_pad_len = actual_q_blocks * block_size - q_len
    k_pad_len = actual_k_blocks * block_size - k_len

    if q_pad_len > 0:
        query_states = F.pad(query_states, (0, 0, 0, q_pad_len), value=0)
    if k_pad_len > 0:
        key_states = F.pad(key_states, (0, 0, 0, k_pad_len), value=0)

    # Reshape into blocks: [batch, heads, num_blocks, block_size, head_dim]
    q_blocks = query_states.view(batch_size, num_heads, actual_q_blocks, block_size, head_dim)
    k_blocks = key_states.view(batch_size, num_heads, actual_k_blocks, block_size, head_dim)

    # Pool within each block
    if pool_method == "avg":
        q_pooled = q_blocks.mean(dim=3)  # [batch, heads, q_blocks, head_dim]
        k_pooled = k_blocks.mean(dim=3)  # [batch, heads, k_blocks, head_dim]
    elif pool_method == "max":
        q_pooled = q_blocks.max(dim=3)[0]
        k_pooled = k_blocks.max(dim=3)[0]
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}. Use 'avg' or 'max'.")

    # Compute block-level attention scores
    # [batch, heads, q_blocks, k_blocks]
    block_scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * scale

    # Apply causal mask at block level
    if causal:
        causal_mask = torch.triu(
            torch.ones(actual_q_blocks, actual_k_blocks, dtype=torch.bool, device=block_scores.device),
            diagonal=1
        )
        block_scores = block_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Apply softmax to get attention distribution
    block_scores_softmax = torch.softmax(block_scores, dim=-1)

    # Select blocks: top-p takes priority over top-k
    if top_p is not None:
        block_mask = _get_topp_mask(block_scores_softmax, top_p, causal, actual_q_blocks, actual_k_blocks)
    elif top_k is not None:
        block_mask = _get_topk_mask(block_scores_softmax, top_k, causal, actual_q_blocks, actual_k_blocks)
    else:
        raise ValueError("Either top_k or top_p must be specified")

    # Pad mask to match chunk-aligned block numbers if needed
    if actual_q_blocks < q_block_num or actual_k_blocks < k_block_num:
        padded_scores = torch.zeros(
            batch_size, num_heads, q_block_num, k_block_num,
            dtype=block_scores_softmax.dtype, device=block_scores_softmax.device
        )
        padded_scores[:, :, :actual_q_blocks, :actual_k_blocks] = block_scores_softmax

        padded_mask = torch.zeros(
            batch_size, num_heads, q_block_num, k_block_num,
            dtype=torch.bool, device=block_mask.device
        )
        padded_mask[:, :, :actual_q_blocks, :actual_k_blocks] = block_mask

        return padded_scores, padded_mask

    return block_scores_softmax, block_mask


def _get_topk_mask(block_scores: torch.Tensor, top_k: int, causal: bool,
                   num_q_blocks: int, num_k_blocks: int) -> torch.Tensor:
    """
    Select top-k blocks per row with causal constraint.

    Args:
        block_scores: [batch, heads, q_blocks, k_blocks] - attention scores
        top_k: Number of blocks to select per row
        causal: Whether to apply causal masking
        num_q_blocks: Number of query blocks
        num_k_blocks: Number of key blocks

    Returns:
        mask: [batch, heads, q_blocks, k_blocks] - True = keep
    """
    batch_size, num_heads, _, _ = block_scores.shape
    device = block_scores.device

    mask = torch.zeros(batch_size, num_heads, num_q_blocks, num_k_blocks,
                       dtype=torch.bool, device=device)

    for i in range(num_q_blocks):
        # For causal, can only attend to blocks 0..i
        if causal:
            valid_k = i + 1
            k = min(top_k, valid_k)
        else:
            valid_k = num_k_blocks
            k = min(top_k, valid_k)

        if k > 0:
            # Get scores for valid range
            scores_row = block_scores[:, :, i, :valid_k]  # [batch, heads, valid_k]
            # Get top-k indices
            _, topk_indices = torch.topk(scores_row, k, dim=-1)  # [batch, heads, k]
            # Scatter True to selected positions
            mask[:, :, i, :].scatter_(-1, topk_indices, True)

    return mask


def _get_topp_mask(block_scores: torch.Tensor, top_p: float, causal: bool,
                   num_q_blocks: int, num_k_blocks: int) -> torch.Tensor:
    """
    Select blocks using top-p (nucleus) sampling per row.

    Sort blocks by score descending, accumulate until cumsum > top_p.

    Args:
        block_scores: [batch, heads, q_blocks, k_blocks] - softmax attention scores
        top_p: Cumulative probability threshold (0.0-1.0)
        causal: Whether to apply causal masking
        num_q_blocks: Number of query blocks
        num_k_blocks: Number of key blocks

    Returns:
        mask: [batch, heads, q_blocks, k_blocks] - True = keep
    """
    batch_size, num_heads, _, _ = block_scores.shape
    device = block_scores.device

    mask = torch.zeros(batch_size, num_heads, num_q_blocks, num_k_blocks,
                       dtype=torch.bool, device=device)

    for i in range(num_q_blocks):
        if causal:
            valid_k = i + 1
        else:
            valid_k = num_k_blocks

        if valid_k > 0:
            scores_row = block_scores[:, :, i, :valid_k]  # [batch, heads, valid_k]

            # Sort descending
            sorted_scores, sorted_indices = torch.sort(scores_row, dim=-1, descending=True)

            # Cumulative sum
            cumsum = torch.cumsum(sorted_scores, dim=-1)

            # Find cutoff: first position where cumsum > top_p
            # Include at least 1 block
            cutoff_mask = cumsum <= top_p
            cutoff_mask[..., 0] = True  # Always include at least 1

            # Scatter True to selected positions
            for b in range(batch_size):
                for h in range(num_heads):
                    num_selected = cutoff_mask[b, h].sum().item() + 1  # +1 for the first exceeding
                    num_selected = min(num_selected, valid_k)
                    selected_indices = sorted_indices[b, h, :num_selected]
                    mask[b, h, i, selected_indices] = True

    return mask


def AvgPool_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int = 128,
    chunk_size: int = 16384,
    top_k: int = None,
    top_p: float = None,
    causal: bool = True,
    pool_method: str = "avg",
    **kwargs,
) -> torch.Tensor:
    """
    Prefill attention using AvgPool estimation + block sparse attention.

    Args:
        query_states: Query tensor [batch, num_heads, q_len, head_dim]
        key_states: Key tensor [batch, num_heads, k_len, head_dim]
        value_states: Value tensor [batch, num_heads, k_len, head_dim]
        block_size: Size of each attention block (default: 128)
        chunk_size: Padding alignment (default: 16384)
        top_k: Number of top blocks to select per query block row (mutually exclusive with top_p)
        top_p: Cumulative probability threshold for nucleus sampling (mutually exclusive with top_k)
        causal: Whether to apply causal masking
        pool_method: "avg" for mean pooling, "max" for max pooling

    Returns:
        Attention output [batch, num_heads, q_len, head_dim]
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    # Step 1: Estimate block mask
    block_scores, block_mask = AvgPool_estimate(
        query_states, key_states,
        block_size=block_size,
        chunk_size=chunk_size,
        top_k=top_k,
        top_p=top_p,
        causal=causal,
        pool_method=pool_method,
    )

    # Calculate actual block numbers
    q_block_num = math.ceil(q_len / block_size)
    k_block_num = math.ceil(k_len / block_size)

    # Ensure tensors are on same device
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)

    # Step 2: Prepare tensors for block_sparse_attn_func
    # block_sparse_attn_func expects: [seq_len, num_heads, head_dim]
    assert block_size == 128, f"block_sparse_attn_func requires block_size=128, got {block_size}"
    assert batch_size == 1, f"block_sparse_attn_func requires batch_size=1, got {batch_size}"

    query_states_t = query_states.transpose(1, 2).reshape(q_len, num_heads, head_dim)
    key_states_t = key_states.transpose(1, 2).reshape(k_len, num_heads, head_dim)
    value_states_t = value_states.transpose(1, 2).reshape(k_len, num_heads, head_dim)

    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=query_states.device)
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )

    # Step 3: Execute block sparse attention
    attn_output = block_sparse_attn_func(
        query_states_t,
        key_states_t,
        value_states_t,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        block_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )

    # Step 4: Reshape output back to [batch, num_heads, q_len, head_dim]
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

    return attn_output
