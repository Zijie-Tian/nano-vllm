import torch
import triton
import triton.language as tl

@triton.jit
def _blasst_chunked_prefill_fwd_kernel(
    Q, K, V, sm_scale, threshold_ln_lambda,
    Out, Lse, LseIn, Mask,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lsez, stride_lseh, stride_lsem,
    stride_lsein_z, stride_lsein_h, stride_lsein_m,
    stride_mask_g0, stride_mask_g1, stride_mask_b,
    Z, H, H_KV, N_CTX_Q, N_CTX_K,
    HAS_LSE_IN: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)

    H_i64 = H
    off_z = off_hz // H_i64
    off_h = off_hz % H_i64
    H_KV_i64 = H_KV
    off_h_kv = off_h // (H_i64 // H_KV_i64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    # Pointers
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m_i64[:, None] * stride_qm + offs_d_i64[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h_kv * stride_kh + (offs_n_i64[:, None] * stride_kn + offs_d_i64[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h_kv * stride_vh + (offs_n_i64[:, None] * stride_vn + offs_d_i64[None, :] * stride_vk)

    if HAS_MASK:
        mask_ptrs = Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1
    else:
        mask_ptrs = Mask

    # Load Q
    q_mask = (offs_m[:, None] < N_CTX_Q)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 1. Global state for BLASST Pruning (from history)
    if HAS_LSE_IN:
        lse_in_ptrs = LseIn + off_z * stride_lsein_z + off_h * stride_lsein_h + offs_m_i64 * stride_lsein_m
        m_global = tl.load(lse_in_ptrs, mask=(offs_m < N_CTX_Q), other=-float("inf"))
    else:
        m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # 2. Local state for computing this independent Chunk
    acc_chunk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_chunk = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_chunk = tl.zeros([BLOCK_M], dtype=tl.float32)

    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        # Read input mask
        do_compute = 1
        if HAS_MASK:
            mask_val = tl.load(mask_ptrs + block_idx * stride_mask_b)
            if mask_val == 0:
                do_compute = 0

        # Only process if mask allows it
        if do_compute == 1:
            start_n_aligned = tl.multiple_of(start_n, BLOCK_N)
            k_mask_1d = (offs_n + start_n_aligned) < N_CTX_K
            
            k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            valid_mask = q_mask & k_mask_1d[None, :]
            qk = tl.where(valid_mask, qk, float("-inf"))

            m_local = tl.max(qk, axis=1)

            # skip condition using m_global
            diff = m_local - m_global
            max_diff = tl.max(diff, axis=0)

            # Dynamic BLASST check
            if max_diff < threshold_ln_lambda:
                do_compute = 0
            else:
                # Update global max for FUTURE blocks in this chunk
                m_global = tl.maximum(m_global, m_local)
                
                # Independent Chunk Computation!
                m_chunk_new = tl.maximum(m_chunk, m_local)
                p = tl.math.exp2((qk - m_chunk_new[:, None]) * 1.44269504)
                scale_factor = tl.math.exp2((m_chunk - m_chunk_new) * 1.44269504)
                l_chunk_new = l_chunk * scale_factor + tl.sum(p, axis=1)
                
                v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0)
                acc_chunk = acc_chunk * scale_factor[:, None]
                acc_chunk = acc_chunk + tl.dot(p.to(v.dtype), v)

                m_chunk = m_chunk_new
                l_chunk = l_chunk_new

        # Write back the final decision (combines input mask and dynamic BLASST)
        if HAS_MASK:
            tl.store(mask_ptrs + block_idx * stride_mask_b, tl.cast(do_compute, tl.int8))

        # advance pointers unconditionally
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        block_idx += 1

    # Finalize independent chunk output
    # Avoid NaN if l_chunk is 0 (i.e. entire chunk was skipped)
    l_chunk_safe = tl.where(l_chunk > 0.0, l_chunk, 1.0)
    acc_chunk = acc_chunk / l_chunk_safe[:, None]
    acc_chunk = tl.where(l_chunk[:, None] > 0.0, acc_chunk, 0.0)
    
    lse_chunk = tl.where(l_chunk > 0.0, m_chunk + tl.math.log2(l_chunk_safe) * 0.69314718, -float("inf"))

    # Write output
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m_i64[:, None] * stride_om + offs_d_i64[None, :] * stride_ok)
    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m_i64 * stride_lsem

    tl.store(out_ptrs, acc_chunk.to(Out.dtype.element_ty), mask=q_mask)
    tl.store(lse_ptrs, lse_chunk, mask=(offs_m < N_CTX_Q))


def blasst_chunked_prefill(q, k, v, threshold_ln_lambda=-6.9, 
                           lse_in=None, mask_buffer=None):
    """
    Computes Chunked Prefill Attention with BLASST dynamic pruning and LSE output.
    
    Args:
        q: [batch, num_heads, q_len, head_dim]
        k: [batch, num_heads, kv_len, head_dim]
        v: [batch, num_heads, kv_len, head_dim]
        threshold_ln_lambda: log(lambda) threshold for skipping blocks. Default -6.9.
        lse_in: Optional previous LSE [batch, num_heads, q_len] to guide pruning.
                It is NOT mixed into the output `out` or `lse`, ensuring the output 
                is strictly local to this KV chunk for correct mathematical merging.
        mask_buffer: Optional [grid_0, grid_1, num_blocks] tensor (int8). 
                     Acts as BOTH input (0=skip, 1=compute) and output (records final BLASST decision).
    Returns:
        out: [batch, num_heads, q_len, head_dim] (Strictly local to this KV chunk)
        lse: [batch, num_heads, q_len] (Strictly local to this KV chunk)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    batch, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    
    out = torch.empty_like(q)
    lse = torch.empty((batch, num_heads, q_len), device=q.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(q_len, BLOCK_M), batch * num_heads)
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    has_lse_in = lse_in is not None
    _lse_in = lse_in if has_lse_in else lse
    lsein_strides = _lse_in.stride()
        
    has_mask = mask_buffer is not None
    _mask_buffer = mask_buffer if has_mask else lse
    mask_strides = _mask_buffer.stride()

    _blasst_chunked_prefill_fwd_kernel[grid](
        q, k, v, sm_scale, threshold_ln_lambda,
        out, lse, _lse_in, _mask_buffer,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        lsein_strides[0], lsein_strides[1], lsein_strides[2],
        mask_strides[0], mask_strides[1], mask_strides[2],
        batch, num_heads, num_kv_heads, q_len, kv_len,
        has_lse_in, has_mask,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_dim, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )
    
    return out, lse
