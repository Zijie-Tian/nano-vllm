"""
BLASST Fused Kernel for Efficient Sub-Block Attention

Replaces the nested loop over sub-blocks with a single fused kernel
to reduce kernel launch overhead.
"""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _blasst_chunk_kernel(
    Q, K, V,
    Out, Lse,
    softmax_scale,
    q_stride_0, q_stride_1, q_stride_2,  # batch, head, seq
    kv_stride_0, kv_stride_1, kv_stride_2,
    v_stride_0, v_stride_1, v_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    lse_stride_0, lse_stride_1,
    batch, num_heads, q_len, kv_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention-style kernel for a single KV block.
    Computes attention and returns output + LSE.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_offset = pid_b * q_stride_0 + pid_h * q_stride_1
    kv_offset = pid_b * kv_stride_0 + pid_h * kv_stride_1

    # Load Q
    q_ptrs = Q + q_offset + offs_m[:, None] * q_stride_2 + offs_d[None, :]
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Initialize accumulators
    acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)

    # Loop over KV
    num_blocks = tl.cdiv(kv_len, BLOCK_N)
    for block_idx in range(num_blocks):
        start_n = block_idx * BLOCK_N
        offs_n_block = start_n + offs_n

        # Load K, V
        k_ptrs = K + kv_offset + offs_n_block[:, None] * kv_stride_2 + offs_d[None, :]
        v_ptrs = V + kv_offset + offs_n_block[:, None] * v_stride_2 + offs_d[None, :]

        kv_mask = offs_n_block[:, None] < kv_len
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # QK
        qk = tl.dot(q, tl.trans(k)) * softmax_scale

        # Causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n_block[None, :])
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc_o += tl.dot(p.to(v.dtype), v)

    # Normalize and store
    acc_o = acc_o / l_i[:, None]

    out_ptrs = Out + kv_offset + offs_m[:, None] * out_stride_2 + offs_d[None, :]
    tl.store(out_ptrs, acc_o.to(Out.dtype.element_ty), mask=offs_m[:, None] < q_len)

    lse_ptrs = Lse + (pid_b * num_heads + pid_h) * lse_stride_0 + offs_m
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < q_len)


def blasst_chunk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention for a chunk of queries against a KV block.

    Args:
        q: [batch, num_heads, q_len, head_dim]
        k: [batch, num_heads, kv_len, head_dim]
        v: [batch, num_heads, kv_len, head_dim]
        softmax_scale: Optional scale factor
        causal: Whether to apply causal masking

    Returns:
        out: [batch, num_heads, q_len, head_dim]
        lse: [batch*num_heads, q_len]
    """
    batch, num_heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    lse = torch.empty(batch * num_heads, q_len, dtype=torch.float32, device=q.device)

    BLOCK_M, BLOCK_N, BLOCK_D = 128, 128, head_dim

    grid = (triton.cdiv(q_len, BLOCK_M), num_heads, batch)

    _blasst_chunk_kernel[grid](
        q, k, v, out, lse,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        batch, num_heads, q_len, kv_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )

    return out, lse


@triton.jit
def _merge_attn_kernel(
    O1, LSE1, O2, LSE2,
    Out, LseOut,
    o1_stride_0, o1_stride_1, o1_stride_2,
    o2_stride_0, o2_stride_1, o2_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    lse_stride_0,
    batch_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge two attention outputs using online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    lse1_ptrs = LSE1 + pid_bh * lse_stride_0 + offs_m
    lse2_ptrs = LSE2 + pid_bh * lse_stride_0 + offs_m

    mask = offs_m < seq_len

    lse1 = tl.load(lse1_ptrs, mask=mask, other=float('-inf'))
    lse2 = tl.load(lse2_ptrs, mask=mask, other=float('-inf'))

    m_out = tl.maximum(lse1, lse2)
    exp1 = tl.exp(lse1 - m_out)
    exp2 = tl.exp(lse2 - m_out)
    sum_exp = exp1 + exp2

    o1_ptrs = O1 + pid_bh * o1_stride_0 + offs_m[:, None] * o1_stride_1 + offs_d[None, :]
    o2_ptrs = O2 + pid_bh * o2_stride_0 + offs_m[:, None] * o2_stride_1 + offs_d[None, :]

    o_mask = offs_m[:, None] < seq_len
    o1 = tl.load(o1_ptrs, mask=o_mask, other=0.0)
    o2 = tl.load(o2_ptrs, mask=o_mask, other=0.0)

    out = (o1 * exp1[:, None] + o2 * exp2[:, None]) / sum_exp[:, None]

    out_ptrs = Out + pid_bh * out_stride_0 + offs_m[:, None] * out_stride_1 + offs_d[None, :]
    tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=o_mask)

    lse_out_ptrs = LseOut + pid_bh * lse_stride_0 + offs_m
    tl.store(lse_out_ptrs, m_out + tl.log(sum_exp), mask=mask)


def merge_attn_outputs(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs."""
    batch, num_heads, seq_len, head_dim = o1.shape
    batch_heads = batch * num_heads

    out = torch.empty_like(o1)
    lse_out = torch.empty_like(lse1)

    grid = (triton.cdiv(seq_len, 128), batch_heads)

    _merge_attn_kernel[grid](
        o1, lse1, o2, lse2, out, lse_out,
        o1.stride(0) * o1.stride(1), o1.stride(2), o1.stride(3),
        o2.stride(0) * o2.stride(1), o2.stride(2), o2.stride(3),
        out.stride(0) * out.stride(1), out.stride(2), out.stride(3),
        lse1.stride(0),
        batch_heads, seq_len, head_dim,
        BLOCK_M=128, BLOCK_D=head_dim,
    )

    return out, lse_out
