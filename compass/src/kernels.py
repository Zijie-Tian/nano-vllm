import torch, math
import triton
import triton.language as tl

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
    k_len, # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (chunk_start + (block_id + 1) * block_size - 1) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

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

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

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
    k_len, # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
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

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

@triton.jit
def flat_group_gemm_kernel(Q, K, Out, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_oz, stride_oh, stride_on,
              chunk_start, chunk_end,
              H: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr,
              ):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
        return

    Q_ptrs = Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * stride_qn
    K_ptrs = K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * stride_kn

    Q_ptrs = Q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qn + tl.arange(0, BLOCK_K)[None, :]
    K_ptrs = K_ptrs + tl.arange(0, BLOCK_N)[None, :] * stride_kn + tl.arange(0, BLOCK_K)[:, None]

    num_iters = HEAD_DIM // BLOCK_K
    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(num_iters):
        q = tl.load(Q_ptrs + iter * BLOCK_K)
        k = tl.load(K_ptrs + iter * BLOCK_K)
        o += tl.dot(q, k)

    O_ptrs = Out + batch_id * stride_oz + head_id * stride_oh + block_m * BLOCK_M * stride_on + block_n * BLOCK_N
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]

    tl.store(O_ptrs, o.to(Out.type.element_ty))

@triton.jit
def flat_group_gemm_fuse_reshape_kernel(Q, K, Out, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_oz, stride_oh, stride_on,
              chunk_start, chunk_end,
              H: tl.constexpr,
              STRIDE: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,
              is_caual: tl.constexpr,
              ):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_caual:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    Q_ptrs = Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * STRIDE * stride_qn
    K_ptrs = K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * STRIDE * stride_kn

    Q_ptrs = Q_ptrs + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE) + tl.arange(0, HEAD_DIM)[None, :] + stride_qn * (STRIDE - 1)
    K_ptrs = K_ptrs + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE) + tl.arange(0, HEAD_DIM)[:, None]

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(STRIDE):
        q = tl.load(Q_ptrs - iter * stride_qn)
        k = tl.load(K_ptrs + iter * stride_kn)
        o += tl.dot(q, k)

    O_ptrs = Out + batch_id * stride_oz + head_id * stride_oh + block_m * BLOCK_M * stride_on + block_n * BLOCK_N
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]

    tl.store(O_ptrs, o.to(Out.type.element_ty))


def softmax_fuse_block_sum(attn_weights_slice, reshaped_block_size, segment_size, chunk_start, chunk_end, real_q_len, scale, is_causal=True):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    try:
        assert k_len % segment_size == 0
    except:
        breakpoint()
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = torch.empty((batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size), dtype=attn_weights_slice.dtype, device=attn_weights_slice.device)

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

def flat_group_gemm(query_states, key_states, chunk_start, chunk_end):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    output = torch.empty((batch_size, num_heads, q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (q_len // BLOCK_M, kv_len // BLOCK_N, batch_size * num_heads)
    flat_group_gemm_kernel[grid](
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
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output

def flat_group_gemm_fuse_reshape(query_states, key_states, stride, chunk_start, chunk_end, is_causal=True):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert (key_states.shape[0] == batch_size)
    assert (key_states.shape[1] == num_heads)
    assert (key_states.shape[3] == head_dim)

    output = torch.empty((batch_size, num_heads, q_len // stride, kv_len // stride), dtype=query_states.dtype, device=query_states.device)

    # Adjust block size based on GPU shared memory
    # RTX 3090 has ~100KB, A100/H100 have ~160KB+
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.total_memory < 30 * 1024**3:  # Less than 30GB (e.g., RTX 3090 24GB)
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128
    assert (q_len % (stride * BLOCK_M) == 0)
    assert (kv_len % (stride * BLOCK_N) == 0)

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


# ================================================================
# Stage 1: Compute partial softmax stats for KV chunking
# ================================================================

def softmax_compute_partial_stats(
    attn_weights_slice: torch.Tensor,
    reshaped_block_size: int,
    segment_size: int,
    scale: float,
    chunk_start: int = 0,
    kv_offset: int = 0,
    is_causal: bool = False,
):
    """
    Stage 1: Compute partial softmax statistics for one KV chunk.

    For each query row, computes:
    - m: max value in this chunk
    - l: sum of exp(x - m) in this chunk

    These partial stats can be merged across KV chunks using merge_softmax_stats(),
    then used with softmax_normalize_and_block_sum().

    Args:
        attn_weights_slice: Raw attention scores [batch, heads, q_len, k_chunk_len]
        reshaped_block_size: Block size in reshaped space
        segment_size: Processing segment size
        scale: Softmax scale factor
        chunk_start: Q chunk start position (in reshaped space)
        kv_offset: KV chunk offset (in reshaped space, for causal masking)
        is_causal: Whether to apply causal masking

    Returns:
        Tuple of (m, l) where:
        - m: [batch, heads, q_len] max values per row
        - l: [batch, heads, q_len] partial sums per row
    """
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape

    assert q_len % reshaped_block_size == 0, f"q_len {q_len} not divisible by reshaped_block_size {reshaped_block_size}"
    assert k_len % segment_size == 0, f"k_len {k_len} not divisible by segment_size {segment_size}"

    # Initialize output tensors
    m_out = torch.full(
        (batch_size, num_heads, q_len),
        float('-inf'),
        dtype=torch.float32,
        device=attn_weights_slice.device
    )
    l_out = torch.zeros(
        (batch_size, num_heads, q_len),
        dtype=torch.float32,
        device=attn_weights_slice.device
    )

    # Process each block
    num_blocks = k_len // segment_size

    for block_idx in range(num_blocks):
        k_start = block_idx * segment_size
        k_end = k_start + segment_size

        # Extract this segment
        attn_segment = attn_weights_slice[:, :, :, k_start:k_end]

        # Apply scale BEFORE computing max (matching Triton kernel behavior)
        # Triton kernel: X = tl.load(...).to(tl.float32) * scale
        attn_segment_scaled = attn_segment * scale

        # Compute max and sum for this segment using exp2 (matching Triton kernel)
        # Triton kernel uses exp2 for online softmax
        m_segment = attn_segment_scaled.max(dim=-1, keepdim=True)[0]  # [B, H, q_len, 1]
        l_segment = torch.pow(2.0, attn_segment_scaled - m_segment).sum(dim=-1)  # [B, H, q_len]

        # Causal masking
        if is_causal:
            # For causal attention, only consider K positions <= Q positions
            causal_mask = torch.ones_like(attn_segment, dtype=torch.bool)
            # Position in full sequence
            for q_pos in range(q_len):
                # This Q position can attend to K positions <= q_pos + chunk_start
                global_q_pos = q_pos + chunk_start
                global_k_start = kv_offset + k_start  # K start in this segment
                # In reshaped space, Q at position global_q_pos can attend to K <= global_q_pos
                max_k_rel_pos = global_q_pos - global_k_start
                if max_k_rel_pos >= 0 and max_k_rel_pos < segment_size:
                    causal_mask[:, :, q_pos, max_k_rel_pos+1:] = False
                elif max_k_rel_pos < 0:
                    # Q position is before this KV chunk starts
                    causal_mask[:, :, q_pos, :] = False

            attn_segment_masked = attn_segment_scaled.masked_fill(~causal_mask, float('-inf'))
            m_segment = attn_segment_masked.max(dim=-1, keepdim=True)[0]
            # For l_segment, we need to sum over the masked values
            exp_masked = torch.pow(2.0, attn_segment_masked - m_segment)
            # Zero out positions that are fully masked (all False in causal_mask for this q_pos)
            fully_masked = ~causal_mask.any(dim=-1, keepdim=True)  # [B, H, q_len, 1]
            exp_masked = exp_masked.masked_fill(fully_masked, 0)
            l_segment = exp_masked.sum(dim=-1)  # [B, H, q_len]

        # Online softmax merge
        m_new = torch.maximum(m_out, m_segment.squeeze(-1))

        # Handle the case where m_out or m_segment is -inf (avoid NaN in l_out)
        # When both are -inf, the row is fully masked and l_out should remain 0
        # When m_out is -inf and m_segment is valid, use only l_segment
        # When m_segment is -inf and m_out is valid, use only l_out
        m_out_inf = m_out == float('-inf')
        m_seg_inf = m_segment.squeeze(-1) == float('-inf')

        # Compute scaling factors only for valid (non -inf) values
        scale_out = torch.where(m_out_inf, torch.zeros_like(m_out), torch.pow(2.0, m_out - m_new))
        scale_seg = torch.where(m_seg_inf, torch.zeros_like(m_segment.squeeze(-1)), torch.pow(2.0, m_segment.squeeze(-1) - m_new))

        l_out = l_out * scale_out + l_segment * scale_seg
        m_out = m_new

    return m_out, l_out


# ================================================================
# Stage 2: Merge partial softmax stats from multiple KV chunks
# ================================================================

def merge_softmax_stats(
    m_chunks,
    l_chunks,
):
    """
    Stage 2: Merge partial softmax statistics from multiple KV chunks.

    Uses the online softmax merging formula:
    m_new = max(m1, m2)
    l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)

    Args:
        m_chunks: List of max tensors [batch, heads, q_len] from each chunk
        l_chunks: List of sum tensors [batch, heads, q_len] from each chunk

    Returns:
        Tuple of (m_global, l_global) with same shape as inputs
    """
    assert len(m_chunks) == len(l_chunks), "m_chunks and l_chunks must have same length"
    assert len(m_chunks) > 0, "chunks list cannot be empty"

    # Use log2 scale to match kernel (exp2)
    LOG2E = 1.4426950408889634

    m_global = m_chunks[0].clone()
    l_global = l_chunks[0].clone()

    for i in range(1, len(m_chunks)):
        m_chunk = m_chunks[i]
        l_chunk = l_chunks[i]

        m_new = torch.maximum(m_global, m_chunk)
        # exp2(m - m_new) = 2^(m - m_new)
        l_global = l_global * torch.pow(2.0, m_global - m_new) + l_chunk * torch.pow(2.0, m_chunk - m_new)
        m_global = m_new

    return m_global, l_global


# ================================================================
# Stage 3: Normalize with global stats and compute block sums
# ================================================================

def softmax_normalize_and_block_sum(
    attn_weights_slice: torch.Tensor,
    m_global: torch.Tensor,
    l_global: torch.Tensor,
    reshaped_block_size: int,
    segment_size: int,
    chunk_start: int,
    real_q_len: int,
    scale: float,
    kv_offset: int = 0,
    is_causal: bool = False,
    num_q_blocks: int = None,
):
    """
    Stage 3: Normalize with global stats and compute block-level attention sums.

    Uses pre-computed global m and l (from merge_softmax_stats())
    to correctly normalize softmax values and compute block sums.

    Args:
        attn_weights_slice: Raw attention scores [batch, heads, q_len, k_chunk_len]
        m_global: Global max values [batch, heads, q_len]
        l_global: Global sum values [batch, heads, q_len]
        reshaped_block_size: Block size in reshaped space
        segment_size: Processing segment size
        chunk_start: Start position for this chunk (for masking)
        real_q_len: Actual Q length (before padding)
        scale: Softmax scale factor
        kv_offset: KV chunk offset (in reshaped space, for causal masking)
        is_causal: Whether to apply causal masking
        num_q_blocks: Number of Q blocks to output (if different from input q_len)

    Returns:
        Block-level attention sums [batch, heads, num_q_blocks, k_chunk_blocks]
    """
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape

    assert q_len % reshaped_block_size == 0
    assert k_len % segment_size == 0
    assert segment_size % reshaped_block_size == 0

    # Output shape - use num_q_blocks if provided, otherwise derive from input
    input_q_blocks = q_len // reshaped_block_size
    k_blocks = k_len // reshaped_block_size
    q_blocks = num_q_blocks if num_q_blocks is not None else input_q_blocks

    output = torch.zeros(
        (batch_size, num_heads, q_blocks, k_blocks),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device
    )

    # Process each block row (only for input blocks, up to input_q_blocks)
    for q_block_idx in range(min(q_blocks, input_q_blocks)):
        q_start_local = q_block_idx * reshaped_block_size
        q_end_local = q_start_local + reshaped_block_size

        # Get global stats for this Q block row
        m_row = m_global[:, :, q_start_local:q_end_local]  # [B, H, block_size]
        l_row = l_global[:, :, q_start_local:q_end_local]  # [B, H, block_size]
        attn_row = attn_weights_slice[:, :, q_start_local:q_end_local, :]  # [B, H, block_size, k_len]

        # Compute sum_mask: which Q positions are within real_q_len
        # global_q_pos = chunk_start + q_start_local + [0, 1, ..., block_size-1]
        # sum_mask[i] = (global_q_pos[i] < real_q_len)
        global_q_start = chunk_start + q_start_local
        global_q_end = global_q_start + reshaped_block_size
        sum_mask = torch.ones(
            (reshaped_block_size,), dtype=torch.bool, device=attn_weights_slice.device
        )
        if global_q_end > real_q_len:
            # Some positions are beyond real_q_len, mask them out
            for q_pos in range(reshaped_block_size):
                global_q_pos = global_q_start + q_pos
                if global_q_pos >= real_q_len:
                    sum_mask[q_pos] = False

        # Compute softmax for each segment
        for seg_idx in range(k_len // segment_size):
            seg_start = seg_idx * segment_size
            seg_end = seg_start + segment_size
            attn_seg = attn_row[:, :, :, seg_start:seg_end]  # [B, H, block_size, seg_size]

            # Calculate segment's global K start for causal check
            segment_global_k_start = seg_start + kv_offset

            # Normalize with global stats
            # Match Triton kernel behavior: exp2((attn - m_global) * scale) / l_global
            # The kernel computes: exp2(X * scale - m_i) * l_i_inv where l_i_inv = 1/l_i
            # Note: The kernel applies scale BEFORE computing m_i, so we need to match that
            # The kernel's m_i is computed from (X * scale), so we need to apply scale here too
            attn_scaled = attn_seg * scale  # Apply scale like the kernel does
            attn_shifted = attn_scaled - m_row.unsqueeze(-1)  # [B, H, block_size, seg_size]
            # Use exp2 to match Triton kernel (which uses exp2)
            attn_exp = torch.pow(2.0, attn_shifted)
            # Divide by l_global (same as multiply by l_i_inv in kernel)
            attn_normalized = attn_exp / l_row.unsqueeze(-1)  # [B, H, block_size, seg_size]

            # Apply causal mask
            if is_causal:
                causal_mask = torch.ones_like(attn_seg, dtype=torch.bool)
                for q_pos in range(reshaped_block_size):
                    # Global Q position (in reshaped space)
                    global_q_pos = q_start_local + q_pos + chunk_start
                    # Can only attend to K positions <= global_q_pos
                    # max_k_rel is the max valid K position relative to segment_global_k_start
                    max_k_rel = global_q_pos - segment_global_k_start
                    if max_k_rel >= 0 and max_k_rel < segment_size:
                        # Only positions 0 to max_k_rel are valid
                        causal_mask[:, :, q_pos, max_k_rel+1:] = False
                    elif max_k_rel < 0:
                        # Q position is before this segment starts
                        causal_mask[:, :, q_pos, :] = False
                    # else: max_k_rel >= segment_size, whole segment is valid

                attn_normalized = attn_normalized.masked_fill(~causal_mask, 0)

            # Apply sum_mask to zero out positions beyond real_q_len
            # This matches Triton kernel: sum_mask = offs_q[:, None] < real_q_len
            if global_q_end > real_q_len:
                # Apply mask: positions where sum_mask=False should contribute 0
                sum_mask_expanded = sum_mask.view(1, 1, reshaped_block_size, 1)
                attn_normalized = attn_normalized * sum_mask_expanded

            # Compute block sums
            # Reshape to [B, H, block_size, seg_k_blocks, block_size] and sum over last two dims
            seg_k_blocks = segment_size // reshaped_block_size
            attn_normalized_reshaped = attn_normalized.view(
                batch_size, num_heads, reshaped_block_size, seg_k_blocks, reshaped_block_size
            )
            block_sum = attn_normalized_reshaped.sum(dim=-1).sum(dim=-2)  # [B, H, seg_k_blocks]

            # Add to output at correct K block position
            # Note: k_block_start is LOCAL to this KV chunk's output
            # The output tensor has shape [B, H, q_blocks, k_blocks] where k_blocks is for this chunk
            # Concatenation of chunk outputs happens in the caller (_compute_q_chunk_attention)
            k_block_start = seg_idx * seg_k_blocks
            output[:, :, q_block_idx, k_block_start:k_block_start + seg_k_blocks] += block_sum

    return output
