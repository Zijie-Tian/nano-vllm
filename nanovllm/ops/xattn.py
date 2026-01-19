"""
XAttention block importance estimation with Triton kernels.

Ported from COMPASS project (compass/src/Xattention.py, kernels.py, utils.py).

This module implements the ESTIMATE phase of XAttention, which identifies
important blocks using stride-interleaved Q/K reshaping and Triton kernels.

Architecture:
    XAttention = Estimate (Triton) + Compute (BSA)
    This module: Estimate only
    BSA library: block_sparse_attn (external dependency for compute)

Key functions:
- xattn_estimate: Estimate block importance and generate sparse mask
- flat_group_gemm_fuse_reshape: Fused stride reshape + GEMM kernel
- softmax_fuse_block_sum: Online softmax + block-wise sum kernel
- find_blocks_chunked: Block selection based on cumulative threshold
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


# ============================================================
# Triton Kernels
# ============================================================

@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by segment_size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Fused softmax + block sum kernel with causal masking.

    This kernel performs online softmax on attention weights and sums
    within each block, producing block-level attention scores.

    Algorithm:
    1. Two-pass online softmax (compute max, then normalize)
    2. Apply causal mask (future positions get -inf)
    3. Reshape to blocks and sum within each block

    Args (via grid):
        block_id: Current Q block index
        head_id: Attention head index
        batch_id: Batch index

    Input shape: [batch, heads, q_len, k_len]
    Output shape: [batch, heads, q_blocks, k_blocks]
    """
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (chunk_start + (block_id + 1) * block_size - 1) // segment_size

    # Online softmax state
    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0  # running sum

    # Input pointer setup
    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    # Output pointer setup
    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    # Pass 1: Compute global max and sum (before causal boundary)
    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    # Pass 1 continued: Handle causal boundary
    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    # Pass 2: Normalize and compute block sums (before causal boundary)
    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    # Pass 2 continued: Handle causal boundary
    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    # Pass 2 continued: Zero out future blocks
    for iter in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len,  # we assume k_len is divisible by segment_size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Fused softmax + block sum kernel without causal masking.

    Same as causal version but without causal mask application.
    """
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    # Pass 1: Compute global max and sum
    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    # Pass 2: Normalize and compute block sums
    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def flat_group_gemm_fuse_reshape_kernel(
    Q, K, Out,
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_oz, stride_oh, stride_on,
    chunk_start, chunk_end,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Fused stride reshape + GEMM kernel.

    This kernel computes Q_reshaped @ K_reshaped^T without explicitly
    creating the reshaped tensors, saving memory and bandwidth.

    Stride reshape (inverse mode):
    - K: concat([K[:,:,k::stride,:] for k in range(stride)])
    - Q: concat([Q[:,:,(stride-1-q)::stride,:] for q in range(stride)])

    The kernel simulates this by adjusting pointer arithmetic:
    - Q samples backwards: Q_ptrs starts at (stride-1), steps by -1
    - K samples forwards: K_ptrs starts at 0, steps by +1
    - Both accumulate across stride iterations

    Args (via grid):
        block_m: Q block index (in reshaped space)
        block_n: K block index (in reshaped space)
        batch_id * H + head_id: Combined batch and head index

    Input shapes:
        Q: [batch, heads, q_len, head_dim]
        K: [batch, heads, k_len, head_dim]

    Output shape: [batch, heads, q_len/stride, k_len/stride]
    """
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    # Early exit for causal: skip blocks where K is entirely in the future
    if is_causal:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    # Q pointer: sample from (stride-1) position, step backwards
    Q_ptrs = Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * STRIDE * stride_qn
    Q_ptrs = Q_ptrs + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE) + tl.arange(0, HEAD_DIM)[None, :] + stride_qn * (STRIDE - 1)

    # K pointer: sample from 0 position, step forwards
    K_ptrs = K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * STRIDE * stride_kn
    K_ptrs = K_ptrs + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE) + tl.arange(0, HEAD_DIM)[:, None]

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Accumulate Q @ K^T across stride positions
    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)  # Q steps backwards
        k = tl.load(K_ptrs + iter * stride_kn)  # K steps forwards
        o += tl.dot(q, k)

    # Store output
    O_ptrs = Out + batch_id * stride_oz + head_id * stride_oh + block_m * BLOCK_M * stride_on + block_n * BLOCK_N
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]

    tl.store(O_ptrs, o.to(Out.type.element_ty))


# ============================================================
# Triton Kernel Wrappers
# ============================================================

def softmax_fuse_block_sum(
    attn_weights_slice: torch.Tensor,
    reshaped_block_size: int,
    segment_size: int,
    chunk_start: int,
    chunk_end: int,
    real_q_len: int,
    scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Compute softmax and block-wise sum of attention weights.

    This function takes raw QK^T scores (after stride reshape),
    applies softmax, and sums within each block to produce
    block-level attention scores.

    Args:
        attn_weights_slice: Raw attention scores [batch, heads, q_len, k_len]
        reshaped_block_size: Block size in reshaped space (block_size / stride)
        segment_size: Processing segment size
        chunk_start: Start position for this chunk
        chunk_end: End position for this chunk
        real_q_len: Actual Q length (before padding)
        scale: Softmax scale factor (includes 1/sqrt(d) and stride normalization)
        is_causal: Whether to apply causal masking

    Returns:
        Block-level attention sums [batch, heads, q_blocks, k_blocks]
    """
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape

    assert q_len % reshaped_block_size == 0, f"q_len {q_len} must be divisible by reshaped_block_size {reshaped_block_size}"
    assert k_len % segment_size == 0, f"k_len {k_len} must be divisible by segment_size {segment_size}"
    assert segment_size % reshaped_block_size == 0, f"segment_size {segment_size} must be divisible by reshaped_block_size {reshaped_block_size}"
    assert attn_weights_slice.stride(-1) == 1, "Last dimension must be contiguous"

    output = torch.empty(
        (batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device
    )

    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output


def flat_group_gemm_fuse_reshape(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    stride: int,
    chunk_start: int,
    chunk_end: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Compute fused stride reshape + GEMM for Q @ K^T.

    This is the core estimation kernel of XAttention. It computes
    attention scores between strided Q and K without explicitly
    creating the reshaped tensors.

    The stride reshape (inverse mode) works as:
    - K_reshaped: concat([K[:,:,k::stride,:] for k in range(stride)])
    - Q_reshaped: concat([Q[:,:,(stride-1-q)::stride,:] for q in range(stride)])

    Result: Q_reshaped @ K_reshaped^T with shape [batch, heads, q_len/stride, k_len/stride]

    Args:
        query_states: Q tensor [batch, heads, q_len, head_dim]
        key_states: K tensor [batch, heads, k_len, head_dim]
        stride: Stride for reshape (typically 8)
        chunk_start: Start position (in reshaped space) for causal masking
        chunk_end: End position (in reshaped space) for causal masking
        is_causal: Whether to apply causal masking (skip future blocks)

    Returns:
        Attention scores [batch, heads, q_len/stride, k_len/stride]
    """
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device
    )

    # Adjust block size based on GPU shared memory
    # RTX 3090 has ~100KB, A100/H100 have ~160KB+
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.total_memory < 30 * 1024**3:  # Less than 30GB (e.g., RTX 3090 24GB)
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128

    assert q_len % (stride * BLOCK_M) == 0, f"q_len {q_len} must be divisible by stride*BLOCK_M {stride * BLOCK_M}"
    assert kv_len % (stride * BLOCK_N) == 0, f"kv_len {kv_len} must be divisible by stride*BLOCK_N {stride * BLOCK_N}"

    grid = (q_len // stride // BLOCK_M, kv_len // stride // BLOCK_N, batch_size * num_heads)

    flat_group_gemm_fuse_reshape_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output


# ============================================================
# Block Selection Utilities
# ============================================================

def find_blocks_chunked(
    input_tensor: torch.Tensor,
    current_index: int,
    threshold: float,
    num_to_choose: Optional[int],
    decoding: bool,
    mode: str = "both",
    causal: bool = True,
) -> torch.Tensor:
    """
    Select important blocks based on cumulative attention threshold.

    This function takes block-level attention scores and selects blocks
    that cumulatively account for a specified fraction of total attention.

    Algorithm:
    1. Compute total attention per query block
    2. Sort blocks by attention score (descending)
    3. Accumulate until reaching threshold * total
    4. Mark accumulated blocks as selected
    5. Always keep diagonal blocks (for causal) and sink block

    Args:
        input_tensor: Block attention scores [batch, heads, q_blocks, k_blocks]
        current_index: Current chunk's starting block index
        threshold: Cumulative attention threshold (e.g., 0.9 = keep 90% attention mass)
        num_to_choose: Alternative to threshold - select fixed number of blocks
        decoding: Whether in decode mode (vs prefill)
        mode: "prefill", "decode", or "both"
        causal: Whether to apply causal masking

    Returns:
        Boolean mask [batch, heads, q_blocks, k_blocks] indicating selected blocks
    """
    assert threshold is None or num_to_choose is None, "Only one of threshold or num_to_choose can be specified"

    batch_size, head_num, chunk_num, block_num = input_tensor.shape

    # Special case: prefill mode during decoding - return all True
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)

    # Special case: decode mode during prefill
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask

    # Convert to float for numerical operations
    input_tensor = input_tensor.to(torch.float32)

    if threshold is not None:
        # Compute required cumulative sum
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(torch.float32)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                (batch_size, head_num, chunk_num, 1)
            ).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold

        if causal:
            # Initialize mask with mandatory blocks
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = True  # Sink block always selected

            # Diagonal blocks (current chunk's causal positions)
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )

            # Mask out mandatory blocks for sorting
            other_values = input_tensor.masked_fill(mask, 0)
            sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)

            # Prepend mandatory blocks' contribution
            sorted_values = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            # Get sorted indices (mandatory blocks get high priority)
            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )

            # Compute cumulative sum (excluding current block)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            # Select blocks until threshold is reached
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)

            # Flatten for scatter operation
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)

            # Mark selected blocks
            mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
        else:
            # Non-causal: simple threshold-based selection
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(input_tensor, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)

            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros((batch_size, head_num, chunk_num, 1), device=input_tensor.device),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)

            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[:, torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1), index] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("Block num selection (num_to_choose) not implemented")

    # Enforce causal: zero out future blocks
    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False

    # Validation
    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor, dtype=bool, device=input_tensor.device)
            lambda_mask[:, :, :, 0] = True
            lambda_mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=lambda_mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            assert torch.where(lambda_mask, mask, True).all()

    return mask


def create_causal_mask(
    batch_size: int,
    head_num: int,
    block_size: int,
    block_num: int,
    divide_block_num: int,
) -> torch.Tensor:
    """
    Create a causal attention mask for block-level attention.

    Args:
        batch_size: Batch size
        head_num: Number of attention heads
        block_size: Tokens per block
        block_num: Total number of blocks
        divide_block_num: Block index at which causality boundary is applied

    Returns:
        Causal mask [batch, heads, block_size, block_size * block_num]
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)

    # Mask future blocks
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    # Apply triangular mask at causality boundary
    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask


# ============================================================
# Main Estimation Function
# ============================================================

def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size: int = 128,
    stride: int = 8,
    norm: float = 1.0,
    threshold: float = 0.9,
    chunk_size: int = 16384,
    use_triton: bool = True,
    causal: bool = True,
    keep_sink: bool = False,
    keep_recent: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate block importance for XAttention sparse selection.

    This function implements the estimation phase of XAttention:
    1. Stride-interleaved reshape of Q and K (inverse mode)
    2. Compute block-level attention scores via Triton kernels
    3. Select important blocks based on cumulative threshold

    The result is a boolean mask indicating which K blocks each Q block
    should attend to. This mask can be used with BSA (block_sparse_attn)
    for efficient sparse attention computation.

    Args:
        query_states: Q tensor [batch, heads, q_len, head_dim]
        key_states: K tensor [batch, heads, k_len, head_dim]
        block_size: Block size in tokens (must be 128 for BSA compatibility)
        stride: Stride for Q/K reshape (typically 8)
        norm: Normalization factor for attention scores
        threshold: Cumulative attention threshold (0.0-1.0)
        chunk_size: Processing chunk size for memory efficiency
        use_triton: Whether to use Triton kernels (requires SM 80+)
        causal: Whether to apply causal masking
        keep_sink: Always keep first block (sink tokens)
        keep_recent: Always keep diagonal blocks (recent context)

    Returns:
        attn_sums: Block-level attention scores [batch, heads, q_blocks, k_blocks]
        simple_masks: Boolean mask for sparse attention [batch, heads, q_blocks, k_blocks]

    Example:
        >>> q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
        >>> k = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
        >>> attn_sums, mask = xattn_estimate(q, k, block_size=128, stride=8, threshold=0.9)
        >>> # mask can be used with block_sparse_attn_func for sparse computation
    """
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head, "GQA not supported in estimation (heads must match)"

    # Compute padding to align with chunk_size
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num

    # Pad K and Q if needed
    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to("cuda")
    else:
        pad_query_states = query_states

    # Check GPU capability for Triton
    if use_triton:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major < 8:
            use_triton = False
            print(f"Triton kernel requires SM 80+, got SM {props.major}{props.minor}. Falling back to PyTorch.")

    # Compute reshaped dimensions
    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

    # Non-Triton fallback: explicit reshape
    if not use_triton:
        # K reshape: concat([K[:,:,k::stride,:] for k in range(stride)])
        reshaped_key = torch.cat(
            [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
        )
        # Q reshape (inverse): concat([Q[:,:,(stride-1-q)::stride,:] for q in range(stride)])
        reshaped_query = torch.cat(
            [(pad_query_states[:, :, (stride - 1 - q)::stride, :]) for q in range(stride)],
            dim=-1,
        )

    attn_sum_list = []
    simple_mask_list = []

    # Process each Q chunk
    for chunk_idx in range(q_chunk_num):
        if use_triton:
            # Triton path: fused reshape + GEMM
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size) * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size) * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                is_causal=causal,
            )

            # Fused softmax + block sum
            # Scale factor: log2(e) / sqrt(head_dim) / stride / norm
            # log2(e) ≈ 1.4426950408889634
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            # PyTorch fallback path
            chunked_query = reshaped_query[
                :, :,
                chunk_idx * reshaped_chunk_size : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size),
                :,
            ]

            # Compute attention scores
            attn_weights_slice = torch.matmul(
                chunked_query, reshaped_key.transpose(2, 3)
            ).to("cuda")
            attn_weights_slice = attn_weights_slice / math.sqrt(head_dim) / stride / norm

            # Apply causal mask
            if causal:
                offset_token_chunk_num = k_chunk_num - q_chunk_num
                causal_mask = torch.zeros(
                    (batch_size, num_q_head, reshaped_chunk_size, reshaped_chunk_size * k_chunk_num),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(1, num_q_head, reshaped_chunk_size, reshaped_chunk_size, device=key_states.device) * float("-inf"),
                    diagonal=1,
                )
                if chunk_idx == q_chunk_num - 1 and q_num_to_pad // stride != 0:
                    causal_mask[:, :, (-(q_num_to_pad // stride)):, :] = float("-inf")
                causal_mask[:, :, :, chunk_end:] = float("-inf")
                attn_weights_slice = attn_weights_slice + causal_mask

            # Softmax
            attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(pad_query_states.dtype)

            if chunk_idx == q_chunk_num - 1 and q_num_to_pad // stride != 0:
                attn_weights_slice[:, :, (-(q_num_to_pad // stride)):, :] = 0

            # Block sum
            attn_sum = (
                attn_weights_slice.view(
                    batch_size, num_kv_head, num_blocks_per_chunk, reshaped_block_size, -1, reshaped_block_size
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to("cuda")
            )

        # Select blocks based on threshold
        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key

    # Concatenate results from all chunks
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    # Apply causal mask to final output
    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(torch.ones(q_block_num, q_block_num, dtype=bool, device=key_states.device), diagonal=0),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )

    # Always keep sink block
    if keep_sink:
        simple_masks[:, :, :, 0] = True

    # Always keep diagonal (recent) blocks
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = eye_matrix.unsqueeze(0).unsqueeze(0).expand(1, num_kv_head, q_block_num, q_block_num)
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks


def compute_sparsity(mask: torch.Tensor, causal: bool = True) -> float:
    """
    Compute the sparsity ratio of a block mask.

    Args:
        mask: Boolean mask [batch, heads, q_blocks, k_blocks]
        causal: Whether mask is causal (only lower triangle counts)

    Returns:
        Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    """
    batch, heads, q_blocks, k_blocks = mask.shape

    if causal:
        # Only count lower triangle
        causal_mask = torch.tril(torch.ones(q_blocks, k_blocks, device=mask.device, dtype=torch.bool))
        total_blocks = causal_mask.sum().item() * batch * heads
        selected_blocks = (mask & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
    else:
        total_blocks = mask.numel()
        selected_blocks = mask.sum().item()

    return 1.0 - (selected_blocks / total_blocks)
