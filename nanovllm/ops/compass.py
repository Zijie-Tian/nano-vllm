import math
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _compass_jagged_fwd_kernel(
    Q,
    K_packed,
    V_packed,
    kv_offsets,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    nheads_kv,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    gqa_ratio = nheads // nheads_kv
    off_h_kv = off_h // gqa_ratio

    off_b_kv_idx = off_b * nheads_kv + off_h_kv
    start_n_offset = tl.load(kv_offsets + off_b_kv_idx)
    end_n_offset = tl.load(kv_offsets + off_b_kv_idx + 1)
    seqlen_k = end_n_offset - start_n_offset

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m_i64[:, None] * stride_qm + offs_d_i64[None, :])
    )
    
    k_ptrs = (
        K_packed
        + (start_n_offset + offs_n_i64)[:, None] * stride_kn
        + offs_d_i64[None, :] * stride_kd
    )
    v_ptrs = (
        V_packed
        + (start_n_offset + offs_n_i64)[:, None] * stride_vn
        + offs_d_i64[None, :] * stride_vd
    )
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d_i64[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m_i64[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m_i64[:, None] < seqlen_q) & (offs_d_i64[None, :] < headdim),
                other=0.0,
            )

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        n_remaining = seqlen_k - start_n
        is_even_n = n_remaining >= BLOCK_N

        if is_even_n:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d_i64[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n_i64[:, None] < n_remaining, other=0.0)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn, 
                    mask=(offs_n_i64[:, None] < n_remaining) & (offs_d_i64[None, :] < headdim), 
                    other=0.0
                )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= softmax_scale
        
        if not is_even_n:
            qk += tl.where(offs_n_i64[None, :] < n_remaining, 0, float("-inf"))
        
        if IS_CAUSAL:
            qk += tl.where(
                offs_m_i64[:, None] >= (start_n + offs_n_i64)[None, :], 0, float("-inf")
            )

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, 1)
        l_new = l_i * alpha + l_ij
        acc_o = acc_o * alpha[:, None]

        if is_even_n:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d_i64[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n_i64[:, None] < n_remaining, other=0.0)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn, 
                    mask=(offs_n_i64[:, None] < n_remaining) & (offs_d_i64[None, :] < headdim), 
                    other=0.0
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        m_i = m_new
        l_i = l_new

    acc_o = acc_o / l_i[:, None]
    lse_i = m_i + tl.log(l_i)

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m_i64
    if EVEN_M:
        tl.store(lse_ptrs, lse_i)
    else:
        tl.store(lse_ptrs, lse_i, mask=offs_m_i64 < seqlen_q)

    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m_i64[:, None] * stride_om + offs_d_i64[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d_i64[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m_i64[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m_i64[:, None] < seqlen_q) & (offs_d_i64[None, :] < headdim)
            )

def compass_jagged_chunked_prefill(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    kv_offsets: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Attention using packed 1D Key/Value tensors with variable sequence lengths.
    
    Args:
        q: [batch, seqlen_q, nheads, headdim]
        k_packed: [total_tokens, headdim] -> Contiguous packed keys
        v_packed: [total_tokens, headdim] -> Contiguous packed values
        kv_offsets: [batch * nheads_kv + 1] -> Boundaries mapping items in packed arrays to specific heads.
        sm_scale: Softmax scaling factor.
        causal: Application of causal mask.
    Returns:
        out: [batch, seqlen_q, nheads, headdim]
        lse: [batch, nheads, seqlen_q]
    """
    batch, seqlen_q, nheads, headdim = q.shape
    nheads_kv = (kv_offsets.shape[0] - 1) // batch

    assert q.stride(-1) == 1
    assert k_packed.stride(-1) == 1
    assert v_packed.stride(-1) == 1

    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    _compass_jagged_fwd_kernel[grid](
        q,
        k_packed,
        v_packed,
        kv_offsets,
        out,
        lse,
        sm_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k_packed.stride(0), k_packed.stride(1),
        v_packed.stride(0), v_packed.stride(1),
        out.stride(0), out.stride(2), out.stride(1),
        nheads,
        nheads_kv,
        seqlen_q,
        seqlen_q_rounded,
        headdim,
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=headdim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out, lse[..., :seqlen_q]
