"""
Xattn_chunked.py - XAttention for Chunked Prefill

This module provides XAttention estimation for chunked prefill scenarios,
where Q is processed in chunks while K accumulates across chunks.

Key difference from Xattention.py:
- Designed for single Q chunk processing at a time
- K can be larger than Q (accumulated from previous chunks)
- Maintains state across chunk boundaries
"""

from compass.src.utils import *
import torch
import math
import sys
import torch.nn.functional as F
from compass.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func

# Module-level density tracker for collecting per-layer density across all layers
_density_tracker = {}


def xattn_estimate_chunked(
    query_states: torch.Tensor,  # Current Q chunk: (B, H, q_chunk_len, D)
    key_states: torch.Tensor,    # Accumulated K: (B, H, k_len, D), k_len >= q_chunk_len
    q_start_pos: int,            # Start position of this Q chunk in the full sequence
    block_size: int,
    stride: int,
    norm: float = 1.0,
    softmax: bool = True,
    threshold: float = 0.9,
    select_mode: str = "inverse",
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    chunk_size: int = 16384,     # Alignment chunk size (same as standard version)
) -> torch.Tensor:
    """
    Estimate attention pattern for a single Q chunk in chunked prefill.

    Args:
        query_states: Current Q chunk, shape (B, H, q_chunk_len, D)
        key_states: Accumulated K from all chunks so far, shape (B, H, k_len, D)
        q_start_pos: Start position of this Q chunk in the full sequence
        block_size: Block size for sparse attention (e.g., 128)
        stride: Downsampling stride for estimation (e.g., 4, 8, 16)
        norm: Normalization factor
        softmax: Whether to apply softmax
        threshold: Threshold for block selection
        select_mode: Query-key pairing mode ("inverse", "slash", etc.)
        use_triton: Whether to use Triton kernels
        causal: Whether to apply causal masking
        kdb: Key downsampling factor
        chunk_size: Alignment chunk size for Triton kernels (default 16384)

    Returns:
        attn_sums: Aggregated attention scores per block
        simple_mask: Boolean mask indicating selected blocks
    """
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    # Store original lengths for valid region tracking
    original_q_len = q_len
    original_k_len = k_len

    # Validate inputs
    assert k_len >= q_len, f"K length ({k_len}) must be >= Q length ({q_len})"
    assert q_start_pos + q_len <= k_len, f"Q end position ({q_start_pos + q_len}) exceeds K length ({k_len})"

    # Chunked prefill expects external chunking - warn if Q is too large
    if q_len > chunk_size:
        import warnings
        warnings.warn(
            f"Q length ({q_len}) exceeds chunk_size ({chunk_size}). "
            f"For chunked prefill, split Q externally and call this function for each chunk.",
            UserWarning
        )

    # Calculate block counts (based on original lengths)
    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    q_start_block = q_start_pos // block_size

    # Check Triton compatibility
    if use_triton:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major < 8:
            use_triton = False
            print(f"Triton requires SM 80+, got SM {props.major}{props.minor}, falling back to PyTorch",
                  file=sys.stderr, flush=True)

    # Pad Q and K based on whether we use Triton or not
    if use_triton:
        # For Triton: pad to chunk_size alignment (same as standard version)
        # This ensures kernel alignment requirements are met
        padded_q_len = ((q_len + chunk_size - 1) // chunk_size) * chunk_size
        padded_k_len = ((k_len + chunk_size - 1) // chunk_size) * chunk_size
    else:
        # For PyTorch fallback: pad to block_size alignment is sufficient
        padded_q_len = q_block_num * block_size
        padded_k_len = k_block_num * block_size

    q_pad = padded_q_len - q_len
    k_pad = padded_k_len - k_len

    if q_pad > 0:
        query_states = F.pad(query_states, (0, 0, 0, q_pad), value=0)
    if k_pad > 0:
        key_states = F.pad(key_states, (0, 0, 0, k_pad), value=0)

    # Reshape dimensions
    reshaped_block_size = block_size // stride
    reshaped_q_len = padded_q_len // stride
    reshaped_k_len = padded_k_len // stride

    # Calculate valid lengths in reshaped space (for masking padding)
    valid_q_reshaped = (original_q_len + stride - 1) // stride  # Ceiling division
    valid_k_reshaped = (original_k_len + stride - 1) // stride

    if use_triton:
        if kdb != 1:
            raise ValueError("use_triton and kdb cannot be used together")

        # Compute chunk boundaries in reshaped space
        chunk_start = q_start_block * reshaped_block_size
        chunk_end = chunk_start + reshaped_q_len  # Padded end for computation
        real_q_len = chunk_start + valid_q_reshaped  # Valid end for masking padding

        # Use Triton kernel for efficient computation
        attn_weights = flat_group_gemm_fuse_reshape(
            query_states,
            key_states,
            stride,
            chunk_start,  # q_start in reshaped space
            chunk_end,    # q_end in reshaped space (padded)
            is_causal=causal,
        )

        # softmax_fuse_block_sum parameters:
        # - chunk_start/chunk_end: for causal boundary calculation
        # - real_q_len: for masking Q padding (kernel uses: sum_mask = offs_q < real_q_len)
        attn_sum = softmax_fuse_block_sum(
            attn_weights,
            reshaped_block_size,
            min(4096, reshaped_block_size),
            chunk_start,
            chunk_end,
            real_q_len,
            1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )

        # Extract only the valid block region (attn_sum is based on padded dimensions)
        # The kernel outputs (B, H, padded_q_blocks, padded_k_blocks), we need (B, H, q_block_num, k_block_num)
        attn_sum = attn_sum[:, :, :q_block_num, :k_block_num]
    else:
        # PyTorch fallback implementation
        # Reshape K: interleave positions and concatenate head dims
        reshaped_key = torch.cat(
            [(key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
        )  # (B, H, k_len/stride, D*stride)

        # Reshape Q based on select_mode
        if select_mode == "inverse" or select_mode == "":
            reshaped_query = torch.cat(
                [(query_states[:, :, (stride - 1 - q) :: (stride * kdb), :]) for q in range(stride)],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_query = torch.cat(
                [(query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        else:
            # Default to inverse mode
            reshaped_query = torch.cat(
                [(query_states[:, :, (stride - 1 - q) :: (stride * kdb), :]) for q in range(stride)],
                dim=-1,
            )

        # Compute attention weights: (B, H, q_len/stride/kdb, k_len/stride)
        attn_weights = torch.matmul(
            reshaped_query, reshaped_key.transpose(2, 3)
        ) / math.sqrt(head_dim) / stride / norm

        # Apply causal mask
        if causal:
            # Create causal mask for this Q chunk
            # Q positions: [q_start_pos, q_start_pos + q_len)
            # K positions: [0, k_len)
            # Q[i] can only attend to K[j] where j <= q_start_pos + i

            reshaped_q_positions = reshaped_q_len // kdb
            causal_mask = torch.zeros(
                (batch_size, num_heads, reshaped_q_positions, reshaped_k_len),
                device=key_states.device,
                dtype=attn_weights.dtype,
            )

            # Mask out padding in K
            if k_pad > 0:
                causal_mask[:, :, :, -(k_pad // stride):] = float("-inf")

            # Mask out future positions
            # For each Q position q_idx (in reshaped space), it can see K up to position:
            # (q_start_pos + q_idx * stride * kdb + stride - 1) // stride
            q_start_reshaped = q_start_pos // stride
            for q_idx in range(reshaped_q_positions):
                # This Q token's position in the full sequence (reshaped)
                q_pos_reshaped = q_start_reshaped + q_idx * kdb
                # It can only see K positions <= q_pos_reshaped
                if q_pos_reshaped + 1 < reshaped_k_len:
                    causal_mask[:, :, q_idx, q_pos_reshaped + 1:] = float("-inf")

            # Handle padding in Q
            if q_pad > 0:
                q_pad_reshaped = q_pad // stride // kdb
                if q_pad_reshaped > 0:
                    causal_mask[:, :, -q_pad_reshaped:, :] = float("-inf")

            attn_weights = attn_weights + causal_mask

        # Apply softmax
        if softmax:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            attn_weights = torch.exp(attn_weights).to(query_states.dtype)

        # Zero out padded Q positions
        if q_pad > 0:
            q_pad_reshaped = q_pad // stride // kdb
            if q_pad_reshaped > 0:
                attn_weights[:, :, -q_pad_reshaped:, :] = 0

        # Aggregate to block level
        # attn_weights: (B, H, q_len/stride/kdb, k_len/stride)
        # -> (B, H, q_block_num, reshaped_block_size/kdb, k_block_num, reshaped_block_size)
        # -> sum over last two dims -> (B, H, q_block_num, k_block_num)
        attn_sum = attn_weights.view(
            batch_size,
            num_heads,
            q_block_num,
            reshaped_block_size // kdb,
            k_block_num,
            reshaped_block_size,
        ).sum(dim=-1).sum(dim=-2)

    # Find blocks that exceed threshold
    simple_mask = find_blocks_chunked(
        attn_sum,
        q_start_block,  # offset for causal mask in find_blocks_chunked
        threshold,
        None,
        decoding=False,
        mode="prefill",
        causal=causal,
    )

    # Apply causal constraint on block level
    if causal:
        # For block-level causal: Q block i can only attend to K blocks j where j <= q_start_block + i
        for q_blk_idx in range(q_block_num):
            q_blk_global = q_start_block + q_blk_idx
            if q_blk_global + 1 < k_block_num:
                simple_mask[:, :, q_blk_idx, q_blk_global + 1:] = False

    return attn_sum, simple_mask


def Xattention_chunked_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_start_pos: int,
    stride: int,
    norm: float = 1.0,
    threshold: float = 0.8,
    block_size: int = 128,
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    layer_id: int = None,
    num_layers: int = 32,
):
    """
    XAttention prefill for a single Q chunk.

    This function computes sparse attention for a Q chunk against accumulated K/V.
    Used in chunked prefill scenarios where the context is processed in chunks.

    Args:
        query_states: Current Q chunk (B, H, q_chunk_len, D)
        key_states: Accumulated K (B, H, k_len, D)
        value_states: Accumulated V (B, H, k_len, D)
        q_start_pos: Start position of this Q chunk
        stride: Estimation stride
        norm: Normalization factor
        threshold: Block selection threshold
        block_size: Sparse attention block size
        use_triton: Use Triton kernels
        causal: Apply causal masking
        kdb: Key downsampling factor
        layer_id: Layer index for logging
        num_layers: Total number of layers

    Returns:
        attn_output: Attention output for this Q chunk
    """
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # Estimate attention pattern
    attn_sums, approx_simple_mask = xattn_estimate_chunked(
        query_states,
        key_states,
        q_start_pos=q_start_pos,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        kdb=kdb,
    )

    # Ensure tensors are on same device
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    # Compute sparse attention using block_sparse_attn
    assert block_size == 128, "block_sparse_attn requires block_size=128"
    assert batch_size == 1, "block_sparse_attn requires batch_size=1"

    # Reshape for block_sparse_attn: (seq_len, num_heads, head_dim)
    q_for_attn = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    k_for_attn = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    v_for_attn = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)

    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=query_states.device)
    head_mask_type = torch.ones(num_heads, device=query_states.device, dtype=torch.int32)

    # Pad mask to match block dimensions
    mask_q_blocks = approx_simple_mask.shape[2]
    mask_k_blocks = approx_simple_mask.shape[3]

    if mask_q_blocks < q_block_num or mask_k_blocks < k_block_num:
        padded_mask = torch.zeros(
            batch_size, num_heads, q_block_num, k_block_num,
            dtype=approx_simple_mask.dtype, device=approx_simple_mask.device
        )
        padded_mask[:, :, :mask_q_blocks, :mask_k_blocks] = approx_simple_mask
        approx_simple_mask = padded_mask

    attn_output = block_sparse_attn_func(
        q_for_attn,
        k_for_attn,
        v_for_attn,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )

    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

    del q_for_attn

    # Calculate and track density
    if causal:
        # For chunked prefill, count valid blocks considering q_start_pos
        q_start_block = q_start_pos // block_size
        total_valid_blocks = 0
        for q_idx in range(q_block_num):
            q_global_idx = q_start_block + q_idx
            # This Q block can see K blocks [0, q_global_idx]
            total_valid_blocks += min(q_global_idx + 1, k_block_num)
        total_valid_blocks *= num_heads
    else:
        total_valid_blocks = q_block_num * k_block_num * num_heads

    selected_blocks = approx_simple_mask[:, :, :q_block_num, :k_block_num].sum().item()
    density = selected_blocks / total_valid_blocks if total_valid_blocks > 0 else 0

    # Track density across layers
    global _density_tracker
    if layer_id is not None:
        if layer_id == 0:
            _density_tracker.clear()
        _density_tracker[layer_id] = density
        if layer_id == num_layers - 1:
            min_density = min(_density_tracker.values())
            min_layer = min(_density_tracker, key=_density_tracker.get)
            print(f"[XAttn-Chunked] Min density: {min_density:.2%} (Layer {min_layer})",
                  file=sys.stderr, flush=True)

    del approx_simple_mask, attn_sums
    return attn_output


# For compatibility: wrapper that handles full prefill by chunking internally
def Xattention_chunked_full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride: int,
    norm: float = 1.0,
    threshold: float = 0.8,
    block_size: int = 128,
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    chunk_size: int = 16384,
    layer_id: int = None,
    num_layers: int = 32,
):
    """
    Full prefill using chunked processing internally.

    This function processes the full Q in chunks, simulating chunked prefill
    behavior for testing and validation purposes.
    """
    batch_size, num_heads, seq_len, head_dim = query_states.shape

    if seq_len <= chunk_size:
        # Single chunk, use regular chunked prefill
        return Xattention_chunked_prefill(
            query_states, key_states, value_states,
            q_start_pos=0,
            stride=stride, norm=norm, threshold=threshold,
            block_size=block_size, use_triton=use_triton,
            causal=causal, kdb=kdb,
            layer_id=layer_id, num_layers=num_layers,
        )

    # Process in chunks
    outputs = []
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = min((chunk_idx + 1) * chunk_size, seq_len)

        # For chunked prefill simulation:
        # - Q chunk is [q_start, q_end)
        # - K is [0, q_end) (accumulated up to current position)
        q_chunk = query_states[:, :, q_start:q_end, :]
        k_chunk = key_states[:, :, :q_end, :]
        v_chunk = value_states[:, :, :q_end, :]

        chunk_output = Xattention_chunked_prefill(
            q_chunk, k_chunk, v_chunk,
            q_start_pos=q_start,
            stride=stride, norm=norm, threshold=threshold,
            block_size=block_size, use_triton=use_triton,
            causal=causal, kdb=kdb,
            layer_id=layer_id if chunk_idx == num_chunks - 1 else None,  # Only log on last chunk
            num_layers=num_layers,
        )
        outputs.append(chunk_output)

    return torch.cat(outputs, dim=2)
