import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
    ],
    key=['N_CTX_Q', 'N_CTX_K'],
)
@triton.jit
def _blasst_chunked_prefill_fwd_kernel(
    Q, K, V, sm_scale, threshold_ln_lambda,
    Out, Lse, Mask,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lsez, stride_lseh, stride_lsem,
    stride_mask_g0, stride_mask_g1, stride_mask_b,
    Z, H, H_KV, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)

    H_i64 = H
    off_z = off_hz // H_i64
    off_h = off_hz % H_i64

    # GQA/MQA support
    H_KV_i64 = H_KV
    off_h_kv = off_h // (H_i64 // H_KV_i64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    # Pre-compute fixed offsets
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m_i64[:, None] * stride_qm + offs_d_i64[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h_kv * stride_kh + (offs_n_i64[:, None] * stride_kn + offs_d_i64[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h_kv * stride_vh + (offs_n_i64[:, None] * stride_vn + offs_d_i64[None, :] * stride_vk)

    # mask_ptrs: [grid_0, grid_1, num_blocks]
    if Mask is not None:
        mask_ptrs = Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1
    else:
        mask_ptrs = None

    # load q
    q_mask = (offs_m[:, None] < N_CTX_Q)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # initialize acc, m_i, l_i
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask_1d = (offs_n + start_n) < N_CTX_K

        # load k
        k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0)
        
        # compute qk
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # mask out of bounds
        qk = tl.where(q_mask & k_mask_1d[None, :], qk, float("-inf"))

        # compute local max
        m_local = tl.max(qk, axis=1)

        # skip condition: check if maximum difference across the block is less than threshold
        diff = m_local - m_i
        max_diff = tl.max(diff, axis=0)

        is_computed = max_diff >= threshold_ln_lambda
        
        # Store mask if buffer is provided: 1 if computed, 0 if skipped
        if mask_ptrs is not None:
            tl.store(mask_ptrs + block_idx * stride_mask_b, is_computed.to(tl.int8))

        if is_computed:
            m_new = tl.maximum(m_i, m_local)
            
            # Fast online softmax updates using log2 and exp2
            # p = exp(qk - m_new) => exp2((qk - m_new) * 1.44269504)
            p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
            
            # scale = exp(m_i - m_new) => exp2((m_i - m_new) * 1.44269504)
            scale_factor = tl.math.exp2((m_i - m_new) * 1.44269504)
            
            l_new = l_i * scale_factor + tl.sum(p, axis=1)
            
            # V_j load
            v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0)
            
            # O_i = O_i * scale_factor + P_ij * V_j
            acc = acc * scale_factor[:, None]
            acc = acc + tl.dot(p.to(v.dtype), v)

            # update m_i, l_i
            m_i = m_new
            l_i = l_new

        # advance pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        block_idx += 1

    # O_chunk = O_i / l_i
    acc = acc / l_i[:, None]
    
    # LSE_chunk = m_i + log(l_i) => m_i + log2(l_i) * ln(2)
    lse = m_i + tl.math.log2(l_i) * 0.69314718

    # Write output
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m_i64[:, None] * stride_om + offs_d_i64[None, :] * stride_ok)
    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m_i64 * stride_lsem

    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)
    tl.store(lse_ptrs, lse, mask=(offs_m < N_CTX_Q))


def blasst_chunked_prefill(q, k, v, threshold_ln_lambda=-6.9, mask_buffer=None):
    """
    Computes Chunked Prefill Attention with BLASST dynamic pruning and LSE output.
    
    Args:
        q: [batch, num_heads, q_len, head_dim]
        k: [batch, num_heads, kv_len, head_dim]
        v: [batch, num_heads, kv_len, head_dim]
        threshold_ln_lambda: log(lambda) threshold for skipping blocks. Default -6.9.
        mask_buffer: Optional [grid_0, grid_1, num_blocks] tensor (int8) to store compute mask.
    Returns:
        out: [batch, num_heads, q_len, head_dim]
        lse: [batch, num_heads, q_len] - Log-Sum-Exp state for composition
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    batch, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    
    assert q.is_contiguous() or q.stride(-1) == 1, "Q must be contiguous"
    assert k.is_contiguous() or k.stride(-1) == 1, "K must be contiguous"
    assert v.is_contiguous() or v.stride(-1) == 1, "V must be contiguous"
    
    out = torch.empty_like(q)
    lse = torch.empty((batch, num_heads, q_len), device=q.device, dtype=torch.float32)
    
    # We pass a placeholder grid function that autotune can use
    grid = lambda META: (triton.cdiv(q_len, META['BLOCK_M']), batch * num_heads)
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    mask_strides = (0, 0, 0)
    if mask_buffer is not None:
        mask_strides = mask_buffer.stride()

    _blasst_chunked_prefill_fwd_kernel[grid](
        q, k, v, sm_scale, threshold_ln_lambda,
        out, lse, mask_buffer,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        mask_strides[0], mask_strides[1], mask_strides[2],
        batch, num_heads, num_kv_heads, q_len, kv_len,
        BLOCK_DMODEL=head_dim,
    )
    
    return out, lse
