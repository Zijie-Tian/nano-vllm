"""
Xattn_chunked.py - XAttention for Chunked Prefill

This module implements XAttention for chunked prefill scenarios, aligned with
nano-vllm's xattn_bsa.py theoretical algorithm.

Key features:
1. Q and K dual-axis chunking (2D chunking) using 3-stage softmax merge
2. Streaming density profiling for long sequences
3. GQA support with Majority Voting aggregation

Reference: 3rdparty/nanovllm/nanovllm/kvcache/sparse/xattn_bsa.py
           3rdparty/nanovllm/nanovllm/ops/xattn.py
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
    softmax_compute_partial_stats,
    merge_softmax_stats,
    softmax_normalize_and_block_sum,
)
from block_sparse_attn import block_sparse_attn_func

# Module-level density tracker
_density_tracker = {}


def xattn_estimate_chunked(
    query_states: torch.Tensor,  # Q: (B, H, q_len, D)
    key_states: torch.Tensor,    # K: (B, H, k_len, D)
    q_start_pos: int = 0,
    block_size: int = 128,
    stride: int = 8,
    norm: float = 1.0,
    threshold: float = 0.9,
    select_mode: str = "inverse",
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    chunk_size: int = 4096,    # Q chunk size
    kv_chunk_size: int = 4096, # K chunk size
) -> torch.Tensor:
    """
    Estimate attention pattern with Q and K dual-axis chunking.

    For long sequences, uses:
    1. Q-axis chunking: split Q into chunks
    2. K-axis chunking: split K into chunks (using 3-stage softmax merge)
    3. For each Q chunk, compute attention against all K chunks, merge using online softmax

    Args:
        query_states: Q tensor, shape (B, H, q_len, D)
        key_states: K tensor, shape (B, H, k_len, D)
        q_start_pos: Start position of Q in full sequence
        block_size: Block size (128)
        stride: Downsampling stride (8)
        norm: Normalization factor
        threshold: Block selection threshold
        select_mode: Query-key pairing mode
        use_triton: Use Triton kernels
        causal: Apply causal masking
        kdb: Key downsampling factor
        chunk_size: Q chunk size
        kv_chunk_size: K chunk size

    Returns:
        attn_sums: Attention scores per block
        simple_mask: Boolean mask for selected blocks
    """
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    original_q_len = q_len
    original_k_len = k_len

    assert k_len >= q_len

    # Check Triton compatibility
    if use_triton:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major < 8:
            use_triton = False

    # If sequences are short, use simple algorithm
    if q_len <= chunk_size and k_len <= kv_chunk_size:
        return _xattn_estimate_simple(
            query_states, key_states, q_start_pos, block_size, stride,
            norm, threshold, select_mode, use_triton, causal, kdb, chunk_size
        )

    # ================================================================
    # 2D Chunking: Q-axis and K-axis
    # ================================================================
    # Strategy: For each Q chunk, compute attention against all K chunks
    # Use 3-stage softmax merge to combine results correctly

    BLOCK_M = 128
    alignment = stride * BLOCK_M

    # Calculate Q chunk count
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    padded_q_len = q_len + q_num_to_pad
    q_chunk_num = padded_q_len // chunk_size

    # Pad Q
    if q_num_to_pad > 0:
        query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0)

    # Calculate valid Q blocks
    q_block_num = (original_q_len + block_size - 1) // block_size

    # Initialize output buffer for block attention sums
    attn_sum_all_chunks = []
    mask_all_chunks = []

    # Process each Q chunk
    for q_chunk_idx in range(q_chunk_num):
        q_start = q_chunk_idx * chunk_size
        q_end = min(q_start + chunk_size, original_q_len)
        Q_chunk = query_states[:, :, q_start:q_end, :]
        actual_q_len = q_end - q_start

        # Compute attention for this Q chunk against full K using KV chunking
        chunk_attn_sum, chunk_mask = _compute_q_chunk_attention(
            Q_chunk, key_states,
            q_start_pos=q_start_pos + q_start,
            q_len=actual_q_len,
            original_q_len=original_q_len,
            block_size=block_size,
            stride=stride,
            threshold=threshold,
            chunk_size=chunk_size,
            kv_chunk_size=kv_chunk_size,
            use_triton=use_triton,
            causal=causal,
        )

        attn_sum_all_chunks.append(chunk_attn_sum)
        mask_all_chunks.append(chunk_mask)

    # Concatenate all chunk results along Q dimension
    if attn_sum_all_chunks:
        full_attn_sum = torch.cat(attn_sum_all_chunks, dim=2)  # [B, H, total_q_blocks, k_blocks]
        full_mask = torch.cat(mask_all_chunks, dim=2)

        # Match standard implementation's output size
        # xattn_estimate pads Q to chunk_size boundary, so output has chunk_size // block_size Q blocks
        total_q_blocks = ((original_q_len + chunk_size - 1) // chunk_size) * (chunk_size // block_size)
        full_attn_sum = full_attn_sum[:, :, :total_q_blocks, :]
        full_mask = full_mask[:, :, :total_q_blocks, :]

        return full_attn_sum, full_mask

    # Fallback
    return _xattn_estimate_simple(
        query_states[:, :, :original_q_len, :], key_states,
        q_start_pos, block_size, stride, norm, threshold,
        select_mode, use_triton, causal, kdb, chunk_size
    )


def _compute_q_chunk_attention(
    q_chunk: torch.Tensor,
    k_full: torch.Tensor,
    q_start_pos: int,
    q_len: int,
    original_q_len: int,
    block_size: int,
    stride: int,
    threshold: float,
    chunk_size: int,
    kv_chunk_size: int,
    use_triton: bool,
    causal: bool,
):
    """
    Compute attention for one Q chunk against full K using 3-stage KV chunking.

    This implements the same algorithm as nanovllm's xattn_bsa.py:
    1. First pass: compute partial softmax stats (m, l) for each KV chunk
    2. Merge: combine all partial stats to get global m and l
    3. Second pass: normalize with global stats and compute block sums
    """
    batch_size, num_heads, _, head_dim = q_chunk.shape
    _, _, k_len, _ = k_full.shape

    # Setup parameters (aligned with xattn_estimate)
    BLOCK_M = 128
    alignment = stride * BLOCK_M

    # Calculate padding and block counts to match xattn_estimate behavior
    # In xattn_estimate, Q padding is done to chunk_size, not alignment
    # K padding is done to chunk_size and produces k_block_num blocks
    k_num_to_pad = ((k_len + block_size - 1) // block_size) * block_size - k_len
    padded_total_k_len = k_len + k_num_to_pad
    k_block_num = padded_total_k_len // block_size

    # For Q, match xattn_estimate's padding behavior:
    # Q is padded to the nearest multiple of chunk_size that fits q_len
    # But we're processing a Q chunk here, so we need to pad to alignment
    # for kernel compatibility, then adjust output to match expected block count
    padded_q_len_kernel = ((q_len + alignment - 1) // alignment) * alignment
    if padded_q_len_kernel != q_len:
        q_pad = padded_q_len_kernel - q_len
        q_chunk = F.pad(q_chunk, (0, 0, 0, q_pad), value=0)

    # Calculate dimensions
    reshaped_block_size = block_size // stride  # 16 for block_size=128, stride=8
    q_reshaped_len = padded_q_len_kernel // stride
    kv_chunk_reshaped = kv_chunk_size // stride

    # q_block_num should match what xattn_estimate produces
    # xattn_estimate pads Q to chunk_size boundary, so:
    #   padded_q_len = ((q_len + chunk_size - 1) // chunk_size) * chunk_size
    #   q_block_num = padded_q_len // block_size
    # When called with a Q chunk that fits in one chunk_size, this gives chunk_size // block_size
    q_chunk_padded_len = ((q_len + chunk_size - 1) // chunk_size) * chunk_size
    q_block_num_total = q_chunk_padded_len // block_size

    # But we also need to know how many real (non-padding) Q blocks there are
    # This is used to mask out padding blocks in the output
    # Use global original_q_len to compute the global real_q_block_num
    # Then compute how many real blocks are in this specific Q chunk
    global_real_q_block_num = (original_q_len + block_size - 1) // block_size
    # This chunk starts at q_start_block, so real blocks in this chunk:
    real_q_block_num = max(0, min(q_block_num_total, global_real_q_block_num - q_start_pos // block_size))

    # Q position in reshaped space
    # In the full sequence, Q starts at q_start_pos in original space
    # In reshaped space, each reshaped_block_size corresponds to one block_size in original
    q_start_block = q_start_pos // block_size
    chunk_start = q_start_block * reshaped_block_size
    chunk_end = chunk_start + q_reshaped_len

    # Use q_block_num_total for kernel output size (includes padding blocks)
    q_block_num = q_block_num_total

    # Softmax scale
    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride
    segment_size = min(4096, reshaped_block_size)

    # ================================================================
    # Step 1: First pass - compute partial stats for all KV chunks
    # ================================================================
    m_chunks = []
    l_chunks = []
    num_kv_chunks = (k_len + kv_chunk_size - 1) // kv_chunk_size

    for kv_chunk_idx in range(num_kv_chunks):
        k_start = kv_chunk_idx * kv_chunk_size
        k_end = min(k_start + kv_chunk_size, k_len)
        K_chunk = k_full[:, :, k_start:k_end, :]

        # Pad K chunk to alignment
        actual_k_len = k_end - k_start
        padded_k_len = ((actual_k_len + alignment - 1) // alignment) * alignment
        if padded_k_len != actual_k_len:
            k_pad = padded_k_len - actual_k_len
            K_chunk = F.pad(K_chunk, (0, 0, 0, k_pad), value=0)

        # Compute raw attention scores
        attn_weights_kv = flat_group_gemm_fuse_reshape(
            q_chunk, K_chunk, stride,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_causal=False,  # Don't apply causal here, handle in stats computation
        )

        # KV offset in reshaped space
        kv_offset_reshaped = kv_chunk_idx * kv_chunk_reshaped

        # Compute partial stats (with causal mask)
        m_partial, l_partial = softmax_compute_partial_stats(
            attn_weights_kv,
            reshaped_block_size,
            segment_size,
            scale,
            chunk_start=chunk_start,
            kv_offset=kv_offset_reshaped,
            is_causal=causal,
        )
        m_chunks.append(m_partial)
        l_chunks.append(l_partial)

        del attn_weights_kv

    # ================================================================
    # Step 2: Merge all partial stats
    # ================================================================
    m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)
    del m_chunks, l_chunks

    # ================================================================
    # Step 3: Second pass - normalize and compute block sums
    # ================================================================
    attn_sum_per_kv = []

    for kv_chunk_idx in range(num_kv_chunks):
        k_start = kv_chunk_idx * kv_chunk_size
        k_end = min(k_start + kv_chunk_size, k_len)
        K_chunk = k_full[:, :, k_start:k_end, :]

        # Pad K chunk to alignment
        actual_k_len = k_end - k_start
        padded_k_len = ((actual_k_len + alignment - 1) // alignment) * alignment
        if padded_k_len != actual_k_len:
            k_pad = padded_k_len - actual_k_len
            K_chunk = F.pad(K_chunk, (0, 0, 0, k_pad), value=0)

        # Recompute attention scores
        attn_weights_kv = flat_group_gemm_fuse_reshape(
            q_chunk, K_chunk, stride,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_causal=False,
        )

        kv_offset_reshaped = kv_chunk_idx * kv_chunk_reshaped

        # Normalize with global stats and compute block sums
        # real_q_len is the boundary in reshaped space beyond which Q positions should be masked
        # For 2D chunking (q_start_pos=0, full K): use global sequence end
        # For streaming (q_start_pos>0, accumulated K): use q_start_pos + q_len
        if q_start_pos == 0:
            # 2D chunking or KV-only chunking: global sequence boundary
            real_q_len_global = (original_q_len + stride - 1) // stride
        else:
            # Streaming: current Q chunk ends at q_start_pos + q_len
            real_q_len_global = (q_start_pos + q_len + stride - 1) // stride
        block_sum_kv = softmax_normalize_and_block_sum(
            attn_weights_kv,
            m_global,
            l_global,
            reshaped_block_size,
            segment_size,
            chunk_start=chunk_start,
            real_q_len=real_q_len_global,
            scale=scale,
            kv_offset=kv_offset_reshaped,
            is_causal=causal,
            num_q_blocks=q_block_num,  # Ensure output matches expected Q block count
        )
        attn_sum_per_kv.append(block_sum_kv)

        del attn_weights_kv

    # ================================================================
    # Step 4: Concatenate block sums and select blocks
    # ================================================================
    attn_sum_concat = torch.cat(attn_sum_per_kv, dim=-1)
    del attn_sum_per_kv, m_global, l_global

    # Calculate current_index for find_blocks_chunked
    # Q starts at q_start_block in the full sequence
    current_index = q_start_block

    mask = find_blocks_chunked(
        attn_sum_concat,
        current_index=current_index,
        threshold=threshold,
        num_to_choose=None,
        decoding=False,
        mode="prefill",
        causal=causal,
    )

    # The attn_sum_concat has shape [B, H, q_blocks_kernel, k_blocks]
    # where q_blocks_kernel = padded_q_len_kernel // block_size
    # We need to trim it to q_block_num_total blocks to match standard implementation's padded size

    # Trim to q_block_num_total blocks (matching standard implementation's padded output size)
    attn_sum_concat = attn_sum_concat[:, :, :q_block_num, :]
    mask = mask[:, :, :q_block_num, :]

    # Apply causal mask post-processing
    # For a single Q chunk, we need to apply causal mask to the diagonal portion
    # The diagonal is determined by q_start_block position
    if causal:
        # The number of Q blocks in the diagonal is q_block_num (padded size of this chunk)
        # These Q blocks start at q_start_block in the full sequence
        # So the causal diagonal is from q_start_block to q_start_block + q_block_num
        diag_start = q_start_block
        diag_size = q_block_num
        # Ensure we don't go out of bounds - mask the Q-K diagonal relationship
        if diag_start + diag_size <= mask.shape[3]:
            # mask shape is [B, H, q_blocks, k_blocks]
            # We need to apply causal mask to the last two dimensions
            for q_idx in range(diag_size):
                for k_idx in range(diag_size):
                    if k_idx > q_idx:  # Above diagonal
                        mask[:, :, q_idx, diag_start + k_idx] = False

    # Zero out padding blocks (blocks beyond real_q_block_num)
    # Standard implementation produces zeros for padding blocks in attn_sum
    # For mask, we keep find_blocks_chunked's output which has diagonal pattern [0, block_idx]
    if real_q_block_num < attn_sum_concat.shape[2]:
        attn_sum_concat[:, :, real_q_block_num:, :] = 0
        # Note: We don't modify mask here because:
        # 1. Standard implementation's mask for padding blocks comes from find_blocks_chunked
        # 2. find_blocks_chunked produces [0, block_idx] pattern even for zero-input blocks
        # 3. But we want exact match, so we accept the mask difference in padding blocks
        # The main issue is Q block 122 (last real block) where attn_sum differs significantly

    return attn_sum_concat, mask


def _xattn_estimate_simple(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    q_start_pos: int,
    block_size: int,
    stride: int,
    norm: float = 1.0,
    threshold: float = 0.9,
    select_mode: str = "inverse",
    use_triton: bool = True,
    causal: bool = True,
    kdb: int = 1,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Simple estimation for short sequences."""
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    original_q_len = q_len
    original_k_len = k_len

    if use_triton:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major < 8:
            use_triton = False

    # Calculate block counts AFTER padding (aligned with xattn_estimate)
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len

    padded_q_len = q_len + q_num_to_pad
    padded_k_len = k_len + k_num_to_pad

    q_block_num = padded_q_len // block_size
    k_block_num = padded_k_len // block_size
    q_start_block = q_start_pos // block_size

    q_pad = padded_q_len - q_len
    k_pad = padded_k_len - k_len

    if q_pad > 0:
        query_states = F.pad(query_states, (0, 0, 0, q_pad), value=0)
    if k_pad > 0:
        key_states = F.pad(key_states, (0, 0, 0, k_pad), value=0)

    reshaped_block_size = block_size // stride
    reshaped_chunk_size = chunk_size // stride
    reshaped_q_len = padded_q_len // stride
    reshaped_k_len = padded_k_len // stride

    # Calculate offset between K and Q chunk counts
    q_chunk_num = padded_q_len // chunk_size
    k_chunk_num = padded_k_len // chunk_size
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    valid_q_reshaped = (original_q_len + stride - 1) // stride

    if use_triton:
        chunk_start = (q_start_pos // chunk_size + offset_token_chunk_num) * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_q_len
        real_q_len = chunk_start + valid_q_reshaped

        attn_weights = flat_group_gemm_fuse_reshape(
            query_states, key_states, stride,
            chunk_start, chunk_end, is_causal=causal,
        )

        attn_sum = softmax_fuse_block_sum(
            attn_weights, reshaped_block_size,
            min(4096, reshaped_block_size),
            chunk_start, chunk_end, real_q_len,
            1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )

        attn_sum = attn_sum[:, :, :q_block_num, :k_block_num]
    else:
        reshaped_key = torch.cat(
            [(key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
        )

        reshaped_query = torch.cat(
            [(query_states[:, :, (stride - 1 - q)::stride, :]) for q in range(stride)],
            dim=-1
        )

        attn_weights = torch.matmul(
            reshaped_query, reshaped_key.transpose(2, 3)
        ) / math.sqrt(head_dim) / stride / norm

        attn_sum = attn_weights.view(
            batch_size, num_heads, q_block_num, reshaped_block_size,
            k_block_num, reshaped_block_size
        ).sum(dim=-1).sum(dim=-2)

    # Calculate current_index for find_blocks_chunked
    q_chunk_num = padded_q_len // chunk_size
    k_chunk_num = padded_k_len // chunk_size
    offset_token_chunk_num = k_chunk_num - q_chunk_num
    current_index = k_block_num - q_block_num + offset_token_chunk_num * (chunk_size // block_size)

    simple_mask = find_blocks_chunked(
        attn_sum, current_index, threshold, None,
        decoding=False, mode="prefill", causal=causal,
    )

    if causal:
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
    chunk_size: int = 4096,
    kv_chunk_size: int = 4096,
):
    """XAttention prefill with Q and K dual-axis chunking."""
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, _, k_len, _ = key_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # Estimate attention pattern
    attn_sums, approx_simple_mask = xattn_estimate_chunked(
        query_states, key_states,
        q_start_pos=q_start_pos,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kv_chunk_size=kv_chunk_size,
    )

    # Ensure tensors on same device
    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)

    # Compute sparse attention
    assert block_size == 128, "block_sparse_attn requires block_size=128"
    assert batch_size == 1, "block_sparse_attn requires batch_size=1"

    q_for_attn = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    k_for_attn = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    v_for_attn = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)

    q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=query_states.device)
    k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=query_states.device)
    head_mask_type = torch.ones(num_heads, device=query_states.device, dtype=torch.int32)

    mask_q_blocks = approx_simple_mask.shape[2]
    mask_k_blocks = approx_simple_mask.shape[3]

    if mask_q_blocks < q_block_num or mask_k_blocks < k_block_num:
        padded_mask = torch.zeros(batch_size, num_heads, q_block_num, k_block_num,
                                  dtype=approx_simple_mask.dtype, device=approx_simple_mask.device)
        padded_mask[:, :, :mask_q_blocks, :mask_k_blocks] = approx_simple_mask
        approx_simple_mask = padded_mask

    attn_output = block_sparse_attn_func(
        q_for_attn, k_for_attn, v_for_attn,
        q_cu_seq_lens, k_cu_seq_lens, head_mask_type, None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len, k_len, p_dropout=0.0, deterministic=True, is_causal=causal,
    )

    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

    # Calculate density
    if causal:
        q_start_block = q_start_pos // block_size
        total_valid_blocks = 0
        for q_idx in range(q_block_num):
            q_global_idx = q_start_block + q_idx
            total_valid_blocks += min(q_global_idx + 1, k_block_num)
        total_valid_blocks *= num_heads
    else:
        total_valid_blocks = q_block_num * k_block_num * num_heads

    selected_blocks = approx_simple_mask[:, :, :q_block_num, :k_block_num].sum().item()
    density = selected_blocks / total_valid_blocks if total_valid_blocks > 0 else 0

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

    return attn_output


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
    chunk_size: int = 4096,
    layer_id: int = None,
    num_layers: int = 32,
    kv_chunk_size: int = 4096,
):
    """Full prefill using chunked processing."""
    batch_size, num_heads, seq_len, head_dim = query_states.shape

    if seq_len <= chunk_size:
        return Xattention_chunked_prefill(
            query_states, key_states, value_states,
            q_start_pos=0, stride=stride, norm=norm, threshold=threshold,
            block_size=block_size, use_triton=use_triton, causal=causal, kdb=kdb,
            layer_id=layer_id, num_layers=num_layers,
            chunk_size=chunk_size, kv_chunk_size=kv_chunk_size,
        )

    outputs = []
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = min((chunk_idx + 1) * chunk_size, seq_len)

        q_chunk = query_states[:, :, q_start:q_end, :]
        k_chunk = key_states[:, :, :q_end, :]
        v_chunk = value_states[:, :, :q_end, :]

        chunk_output = Xattention_chunked_prefill(
            q_chunk, k_chunk, v_chunk,
            q_start_pos=q_start, stride=stride, norm=norm, threshold=threshold,
            block_size=block_size, use_triton=use_triton, causal=causal, kdb=kdb,
            layer_id=layer_id if chunk_idx == num_chunks - 1 else None,
            num_layers=num_layers, chunk_size=chunk_size, kv_chunk_size=kv_chunk_size,
        )
        outputs.append(chunk_output)

    return torch.cat(outputs, dim=2)


def compute_streaming_density_profile(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    stride: int = 8,
    block_size: int = 128,
    threshold: float = 0.95,
    chunk_size: int = 4096,
    kv_chunk_size: int = 4096,
) -> dict:
    """Compute density profile for streaming scenarios."""
    batch_size, num_heads, seq_len, head_dim = query_states.shape

    densities = []
    chunk_densities = []

    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * chunk_size
        q_end = min((chunk_idx + 1) * chunk_size, seq_len)

        q_chunk = query_states[:, :, q_start:q_end, :]
        k_chunk = key_states[:, :, :q_end, :]

        attn_sum, mask = xattn_estimate_chunked(
            q_chunk, k_chunk,
            q_start_pos=q_start,
            block_size=block_size,
            stride=stride,
            threshold=threshold,
            chunk_size=chunk_size,
            kv_chunk_size=kv_chunk_size,
        )

        q_block_num = (q_end - q_start + block_size - 1) // block_size
        k_block_num = (q_end + block_size - 1) // block_size

        q_start_block = q_start // block_size
        valid_blocks = 0
        for qb in range(q_block_num):
            qb_global = q_start_block + qb
            valid_blocks += min(qb_global + 1, k_block_num)

        selected = mask[:, :, :q_block_num, :k_block_num].sum().item()
        chunk_density = selected / (valid_blocks * num_heads) if valid_blocks > 0 else 0

        densities.append(chunk_density)
        chunk_densities.append({
            'chunk_idx': chunk_idx,
            'q_start': q_start,
            'q_end': q_end,
            'density': chunk_density,
        })

    return {
        'per_chunk': chunk_densities,
        'avg_density': sum(densities) / len(densities) if densities else 0,
        'min_density': min(densities) if densities else 0,
        'max_density': max(densities) if densities else 0,
    }


def majority_voting_aggregate(
    masks: torch.Tensor,
    bsa_per_cpu: int = 32,
) -> torch.Tensor:
    """Aggregate masks using majority voting for GQA."""
    B, H, Q_bsa, K_bsa = masks.shape

    num_cpu_blocks = K_bsa // bsa_per_cpu

    if num_cpu_blocks == 1:
        return masks.any(dim=1)

    masks_per_cpu = masks.view(B, H, Q_bsa, num_cpu_blocks, bsa_per_cpu)
    cpu_block_selected = masks_per_cpu.any(dim=-1)
    votes = cpu_block_selected.sum(dim=1)
    threshold = H // 2

    aggregated = votes > threshold

    return aggregated
