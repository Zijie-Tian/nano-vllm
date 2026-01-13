"""
COMPASS: COntextual Multi-head Parallel Attention with Sparse Selection

This module implements the COMPASS sparse attention mechanism for efficient
long-context LLM inference.

Estimation Strategy (based on rowmax sparsity):
- Compute QK matmul to get attention scores per block
- For each query block row, traverse key blocks from left to right
- Track m (global max so far) and m_local (current block's max)
- If (m - m_local) > lambda, mark the block as sparse

Structure:
- Compass_estimate(): Estimates important blocks using rowmax-based filtering
- Compass_prefill(): Uses the estimate to perform block sparse attention
"""

import torch
import math
import torch.nn.functional as F
from block_sparse_attn import block_sparse_attn_func


def Compass_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int = 128,
    causal: bool = True,
    lambd: float = 5.0,
    chunk_size: int = 16384,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    COMPASS block importance estimation using rowmax-based filtering.

    For each query block row, we traverse key blocks and track:
    - m: the running maximum attention score across all blocks seen so far
    - m_local: the maximum attention score in the current block

    A block is marked as sparse if: (m - m_local) > lambda
    This means the block's contribution is negligible compared to more important blocks.

    Args:
        query_states: Query tensor of shape [batch, num_heads, q_len, head_dim]
        key_states: Key tensor of shape [batch, num_heads, k_len, head_dim]
        block_size: Size of each attention block (default: 128)
        causal: Whether to use causal attention mask
        lambd: Threshold for sparsity decision. If (m - m_local) > lambd, block is sparse.
        chunk_size: Chunk size for block number calculation (default: 16384)

    Returns:
        block_rowmax: Maximum attention scores per block [batch, num_heads, q_blocks, k_blocks]
        block_mask: Boolean mask indicating which blocks to compute (True = keep, False = sparse)
                    Shape: [batch, num_heads, q_blocks, k_blocks]
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    # Calculate number of blocks (with padding for chunk alignment)
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len

    k_block_num = (k_len + k_num_to_pad) // block_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    # Actual blocks (without padding)
    actual_k_blocks = math.ceil(k_len / block_size)
    actual_q_blocks = math.ceil(q_len / block_size)

    # Scale factor for attention scores
    scale = 1.0 / math.sqrt(head_dim)

    # Initialize outputs
    block_rowmax = torch.full(
        (batch_size, num_heads, q_block_num, k_block_num),
        float('-inf'),
        dtype=query_states.dtype,
        device=query_states.device
    )
    block_mask = torch.zeros(
        batch_size, num_heads, q_block_num, k_block_num,
        dtype=torch.bool,
        device=query_states.device
    )

    # Process each batch and head
    for b in range(batch_size):
        for h in range(num_heads):
            q_h = query_states[b, h]  # [q_len, head_dim]
            k_h = key_states[b, h]    # [k_len, head_dim]

            for i in range(actual_q_blocks):
                # Get query block
                q_start = i * block_size
                q_end = min((i + 1) * block_size, q_len)
                q_block = q_h[q_start:q_end]  # [block_size_q, head_dim]

                # First pass: compute rowmax for each key block
                # Track m (running maximum across all key blocks)
                m = float('-inf')

                # Determine how many key blocks to process (causal)
                if causal:
                    max_j = i + 1  # Can only attend to blocks 0..i
                else:
                    max_j = actual_k_blocks

                for j in range(max_j):
                    # Get key block
                    k_start = j * block_size
                    k_end = min((j + 1) * block_size, k_len)
                    k_block = k_h[k_start:k_end]  # [block_size_k, head_dim]

                    # Compute QK^T for this block
                    qk = torch.matmul(q_block, k_block.T) * scale  # [block_size_q, block_size_k]

                    # Apply causal mask within diagonal blocks
                    if causal and i == j:
                        # Within the diagonal block, apply token-level causal mask
                        mask = torch.triu(torch.ones_like(qk, dtype=torch.bool), diagonal=1)
                        qk = qk.masked_fill(mask, float('-inf'))

                    # Compute m_local (max in this block)
                    m_local = qk.max().item()
                    block_rowmax[b, h, i, j] = m_local

                    # Update running maximum
                    m = max(m, m_local)

                # Second pass: determine sparsity based on (m - m_local) > lambda
                for j in range(max_j):
                    m_local = block_rowmax[b, h, i, j].item()

                    # Block is kept if difference is within threshold
                    # (m - m_local) <= lambda means block is important
                    if m - m_local <= lambd:
                        block_mask[b, h, i, j] = True
                    # else: block_mask stays False (sparse)

                # Mark future blocks as sparse for causal attention
                if causal:
                    for j in range(max_j, k_block_num):
                        block_mask[b, h, i, j] = False  # Already False, but explicit

    return block_rowmax, block_mask


def Compass_estimate_fast(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int = 128,
    causal: bool = True,
    lambd: float = 5.0,
    chunk_size: int = 16384,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast vectorized version of COMPASS estimation.

    Same algorithm as Compass_estimate but uses batched matrix operations
    for better GPU utilization.

    Args:
        query_states: Query tensor of shape [batch, num_heads, q_len, head_dim]
        key_states: Key tensor of shape [batch, num_heads, k_len, head_dim]
        block_size: Size of each attention block (default: 128)
        causal: Whether to use causal attention mask
        lambd: Threshold for sparsity decision
        chunk_size: Chunk size for block number calculation

    Returns:
        block_rowmax: Maximum attention scores per block [batch, num_heads, q_blocks, k_blocks]
        block_mask: Boolean mask indicating which blocks to compute (True = keep)
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    # Calculate number of blocks
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len

    k_block_num = (k_len + k_num_to_pad) // block_size
    q_block_num = (q_len + q_num_to_pad) // block_size

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

    # Initialize outputs
    block_rowmax = torch.full(
        (batch_size, num_heads, q_block_num, k_block_num),
        float('-inf'),
        dtype=query_states.dtype,
        device=query_states.device
    )

    # Compute block-wise attention scores
    # For each (i, j) block pair, compute max(Q_i @ K_j^T)
    for i in range(actual_q_blocks):
        q_block = q_blocks[:, :, i]  # [batch, heads, block_size, head_dim]

        # Determine range of key blocks (causal)
        max_j = i + 1 if causal else actual_k_blocks

        for j in range(max_j):
            k_block = k_blocks[:, :, j]  # [batch, heads, block_size, head_dim]

            # Compute QK^T: [batch, heads, block_size_q, block_size_k]
            qk = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Apply causal mask within diagonal blocks
            if causal and i == j:
                causal_mask = torch.triu(
                    torch.ones(block_size, block_size, dtype=torch.bool, device=qk.device),
                    diagonal=1
                )
                qk = qk.masked_fill(causal_mask, float('-inf'))

            # Compute rowmax for this block
            m_local = qk.amax(dim=(-2, -1))  # [batch, heads]
            block_rowmax[:, :, i, j] = m_local

    # Compute running maximum along key dimension for each query block
    # m[i] = max(block_rowmax[i, 0:i+1]) for causal
    m_running = torch.full(
        (batch_size, num_heads, q_block_num),
        float('-inf'),
        dtype=query_states.dtype,
        device=query_states.device
    )

    for i in range(actual_q_blocks):
        max_j = i + 1 if causal else actual_k_blocks
        m_running[:, :, i] = block_rowmax[:, :, i, :max_j].amax(dim=-1)

    # Compute sparsity mask: keep if (m - m_local) <= lambda
    # Expand m_running for broadcasting: [batch, heads, q_blocks, 1]
    m_expanded = m_running.unsqueeze(-1)

    # Difference: [batch, heads, q_blocks, k_blocks]
    diff = m_expanded - block_rowmax

    # Block is kept (True) if diff <= lambda
    block_mask = diff <= lambd

    # Apply causal mask (blocks beyond diagonal are always sparse)
    if causal:
        q_indices = torch.arange(q_block_num, device=query_states.device).view(1, 1, -1, 1)
        k_indices = torch.arange(k_block_num, device=query_states.device).view(1, 1, 1, -1)
        causal_block_mask = q_indices >= k_indices
        block_mask = block_mask & causal_block_mask

    return block_rowmax, block_mask


def Compass_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int = 128,
    causal: bool = True,
    lambd: float = 5.0,
    chunk_size: int = None,
    use_fast: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    COMPASS attention prefill using full attention.

    Currently uses full attention (all blocks selected) via block_sparse_attn_func.
    The block_mask is set to include all causal blocks (lower triangular at block level).

    Args:
        query_states: Query tensor of shape [batch, num_heads, q_len, head_dim]
        key_states: Key tensor of shape [batch, num_heads, k_len, head_dim]
        value_states: Value tensor of shape [batch, num_heads, k_len, head_dim]
        block_size: Size of each attention block (default: 128)
        causal: Whether to use causal attention mask
        lambd: (unused) Threshold for sparsity - kept for API compatibility
        chunk_size: (unused) Chunk size - kept for API compatibility
        use_fast: (unused) - kept for API compatibility

    Returns:
        Attention output of shape [batch, num_heads, q_len, head_dim]
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # Create full block mask (all blocks selected)
    # For causal attention: lower triangular at block level (q_block >= k_block)
    if causal:
        q_indices = torch.arange(q_block_num, device=query_states.device).view(1, 1, -1, 1)
        k_indices = torch.arange(k_block_num, device=query_states.device).view(1, 1, 1, -1)
        block_mask = (q_indices >= k_indices).expand(batch_size, num_heads, q_block_num, k_block_num)
    else:
        # Non-causal: all blocks are selected
        block_mask = torch.ones(
            batch_size, num_heads, q_block_num, k_block_num,
            dtype=torch.bool,
            device=query_states.device
        )

    # Ensure all tensors are on the same device
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)

    # Prepare tensors for block_sparse_attn_func
    # block_sparse_attn_func expects: [seq_len, num_heads, head_dim]
    assert block_size == 128, f"block_sparse_attn_func requires block_size=128, got {block_size}"
    assert batch_size == 1, f"block_sparse_attn_func requires batch_size=1, got {batch_size}"

    query_states_t = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    key_states_t = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states_t = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)

    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )

    # Perform block sparse attention
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

    # Reshape output back to [batch, num_heads, q_len, head_dim]
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

    # Cleanup
    del block_mask

    return attn_output
