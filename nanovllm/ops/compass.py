import math
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _compass_jagged_fwd_kernel(
    Q,
    K_packed,
    V_packed,
    kv_offsets,
    Out,
    Lse,
    softmax_scale,
    threshold_ln_lambda,
    MglobalIn,
    MglobalOut,
    Mask,
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
    stride_mgin_b,
    stride_mgin_h,
    stride_mgin_m,
    stride_mgout_b,
    stride_mgout_h,
    stride_mgout_m,
    stride_mask_g0,
    stride_mask_g1,
    stride_mask_b,
    nheads,
    nheads_kv,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    HAS_MGLOBAL_IN: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
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
    
    if HAS_MASK:
        mask_ptrs = Mask + start_m * stride_mask_g0 + off_hb * stride_mask_g1
    else:
        mask_ptrs = Mask

    # Load Q
    q_mask = offs_m_i64[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_mask & (offs_d_i64[None, :] < headdim), other=0.0)

    # Load m_global for BLASST pruning
    if HAS_MGLOBAL_IN:
        mgin_ptrs = (
            MglobalIn
            + off_b * stride_mgin_b
            + off_h * stride_mgin_h
            + offs_m_i64 * stride_mgin_m
        )
        m_global = tl.load(mgin_ptrs, mask=(offs_m < seqlen_q), other=-float("inf"))
    else:
        m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Local state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    
    block_idx = 0
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        n_remaining = seqlen_k - start_n
        is_even_n = n_remaining >= BLOCK_N
        k_mask_1d = offs_n_i64 < n_remaining

        # Load K
        if is_even_n:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d_i64[None, :] < headdim, other=0.0)
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(k_mask_1d[:, None]) & (offs_d_i64[None, :] < headdim),
                other=0.0,
            )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= softmax_scale
        
        if not is_even_n:
            qk += tl.where(k_mask_1d[None, :], 0, float("-inf"))
        
        if IS_CAUSAL:
            qk += tl.where(
                offs_m_i64[:, None] >= (start_n + offs_n_i64)[None, :], 0, float("-inf")
            )

        m_local = tl.max(qk, 1)

        # BLASST skip decision: compare m_local against historical m_global
        diff = m_local - m_global
        max_diff = tl.max(diff, axis=0)

        # Update m_global UNCONDITIONALLY after check
        m_global = tl.maximum(m_global, m_local)

        do_compute = 1
        if max_diff < threshold_ln_lambda:
            do_compute = 0

        if do_compute == 1:
            # Standard online softmax + PV accumulation
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_ij = tl.sum(p, 1)
            l_new = l_i * alpha + l_ij
            acc_o = acc_o * alpha[:, None]

            # Load V
            if is_even_n:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d_i64[None, :] < headdim, other=0.0)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(k_mask_1d[:, None]) & (offs_d_i64[None, :] < headdim),
                    other=0.0,
                )

            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            m_i = m_new
            l_i = l_new

        # Write back mask (combines BLASST dynamic decision)
        if HAS_MASK:
            tl.store(
                mask_ptrs + block_idx * stride_mask_b, tl.cast(do_compute, tl.int8)
            )

        block_idx += 1

    # Finalize
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc_o = acc_o / l_safe[:, None]
    acc_o = tl.where(l_i[:, None] > 0.0, acc_o, 0.0)

    lse_i = tl.where(
        l_i > 0.0, m_i + tl.log(l_safe), -float("inf")
    )

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m_i64
    tl.store(lse_ptrs, lse_i, mask=offs_m_i64 < seqlen_q)

    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m_i64[:, None] * stride_om + offs_d_i64[None, :])
    )
    tl.store(
        out_ptrs, acc_o,
        mask=(offs_m_i64[:, None] < seqlen_q) & (offs_d_i64[None, :] < headdim),
    )

    # Write m_global output
    mgout_ptrs = (
        MglobalOut
        + off_b * stride_mgout_b
        + off_h * stride_mgout_h
        + offs_m_i64 * stride_mgout_m
    )
    tl.store(mgout_ptrs, m_global, mask=(offs_m < seqlen_q))


def compass_jagged_chunked_prefill(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    kv_offsets: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
    threshold_ln_lambda: float = -23.0,
    m_global_in: Optional[torch.Tensor] = None,
    mask_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Attention using packed 1D Key/Value tensors with BLASST L2 pruning.
    
    Args:
        q: [batch, seqlen_q, nheads, headdim]
        k_packed: [total_tokens, headdim] -> Contiguous packed keys
        v_packed: [total_tokens, headdim] -> Contiguous packed values
        kv_offsets: [batch * nheads_kv + 1] -> Boundaries mapping items in packed arrays to specific heads.
        sm_scale: Softmax scaling factor.
        causal: Application of causal mask.
        threshold_ln_lambda: log(lambda) threshold for BLASST skip. Default -23.0 (effectively no skip).
        m_global_in: Optional [batch, nheads, seqlen_q] running max from previous pipeline piece.
        mask_buffer: Optional [grid_0, batch*nheads, max_kv_blocks] int8 tensor.
                     Initialized to 1 by caller. Kernel writes 0 for skipped blocks.
    Returns:
        out: [batch, seqlen_q, nheads, headdim]
        lse: [batch, nheads, seqlen_q]
        m_global_out: [batch, nheads, seqlen_q]
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
    m_global_out = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

    has_mglobal_in = m_global_in is not None
    _mgin = m_global_in if has_mglobal_in else lse  # dummy
    mgin_strides = _mgin.stride()

    has_mask = mask_buffer is not None
    _mask = mask_buffer if has_mask else lse  # dummy
    mask_strides = _mask.stride()

    _compass_jagged_fwd_kernel[grid](
        q,
        k_packed,
        v_packed,
        kv_offsets,
        out,
        lse,
        sm_scale,
        threshold_ln_lambda,
        _mgin,
        m_global_out,
        _mask,
        q.stride(0), q.stride(2), q.stride(1),
        k_packed.stride(0), k_packed.stride(1),
        v_packed.stride(0), v_packed.stride(1),
        out.stride(0), out.stride(2), out.stride(1),
        mgin_strides[0], mgin_strides[1], mgin_strides[2],
        m_global_out.stride(0), m_global_out.stride(1), m_global_out.stride(2),
        mask_strides[0], mask_strides[1], mask_strides[2],
        nheads,
        nheads_kv,
        seqlen_q,
        seqlen_q_rounded,
        headdim,
        has_mglobal_in,
        has_mask,
        causal,
        BLOCK_HEADDIM=headdim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out, lse[..., :seqlen_q], m_global_out


# ============================================================
# In-place Merge Attention Outputs (COMPASS-only optimization)
# ============================================================

@triton.jit
def _merge_attention_inplace_kernel(
    O1,          # [batch, seqlen_q, nheads, headdim] - will be overwritten
    O2,          # [batch, seqlen_q, nheads, headdim]
    Lse1,        # [batch, nheads, seqlen_q] - will be overwritten
    Lse2,        # [batch, nheads, seqlen_q]
    stride_o_b,
    stride_o_s,
    stride_o_h,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    seqlen_q,
    headdim,
    BLOCK_D: tl.constexpr,
):
    """Fused in-place merge: o1 = merge(o1, o2), lse1 = merge(lse1, lse2).

    Each program handles one (batch, seq_q, head) position.
    All arithmetic in fp32 for numerical stability.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Load LSE values (fp32)
    lse_idx = pid_b * stride_lse_b + pid_h * stride_lse_h + pid_s * stride_lse_s
    lse1_val = tl.load(Lse1 + lse_idx).to(tl.float32)
    lse2_val = tl.load(Lse2 + lse_idx).to(tl.float32)

    # Compute scaling factors in fp32
    max_lse = tl.maximum(lse1_val, lse2_val)
    exp1 = tl.exp(lse1_val - max_lse)
    exp2 = tl.exp(lse2_val - max_lse)
    sum_exp = exp1 + exp2

    # Merged LSE = max_lse + log(sum_exp)
    lse_merged = max_lse + tl.log(sum_exp)
    tl.store(Lse1 + lse_idx, lse_merged)

    # Merge output vectors: o1 = (o1 * exp1 + o2 * exp2) / sum_exp
    o_base = pid_b * stride_o_b + pid_s * stride_o_s + pid_h * stride_o_h
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < headdim

    o1_vals = tl.load(O1 + o_base + offs_d, mask=mask, other=0.0).to(tl.float32)
    o2_vals = tl.load(O2 + o_base + offs_d, mask=mask, other=0.0).to(tl.float32)

    o_merged = (o1_vals * exp1 + o2_vals * exp2) / sum_exp
    tl.store(O1 + o_base + offs_d, o_merged, mask=mask)


def merge_attention_inplace(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> None:
    """In-place merge two attention outputs. Results written to o1 and lse1.

    This is a COMPASS-only optimization that fuses LSE merge + output merge
    into a single Triton kernel with zero memory allocation.

    Args:
        o1: First output [batch, seqlen_q, nheads, headdim] — overwritten with merged result
        lse1: First LSE [batch, nheads, seqlen_q] — overwritten with merged result
        o2: Second output [batch, seqlen_q, nheads, headdim]
        lse2: Second LSE [batch, nheads, seqlen_q]
    """
    batch, seqlen_q, nheads, headdim = o1.shape

    grid = (batch, seqlen_q, nheads)
    BLOCK_D = triton.next_power_of_2(headdim)

    _merge_attention_inplace_kernel[grid](
        o1, o2, lse1, lse2,
        o1.stride(0), o1.stride(1), o1.stride(2),
        lse1.stride(0), lse1.stride(1), lse1.stride(2),
        seqlen_q,
        headdim,
        BLOCK_D=BLOCK_D,
    )
