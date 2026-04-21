"""
PostRoPE sparse prefill policy.

This policy keeps the public `POSTROPE` policy name and post-RoPE KV semantics,
but implements a fully self-contained SpargeAttn-style two-stage chunked prefill
path without delegating to other sparse policies:

1. Stage 1 (`select_blocks`): GPU-side pooled-Q / pooled-K scoring with
   self-similarity protection and top-p chunk selection.
2. Stage 2 (`compute_chunked_prefill`): online sparse compute on the selected
   historical chunks plus the current causal chunk.

Decode and GPU-only paths remain semantically full attention, but are
implemented locally in this policy rather than via policy delegation.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from .policy import SparsePolicy

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence
    from nanovllm.kvcache.manager import KVCacheManager
    from nanovllm.kvcache.offload_engine import OffloadEngine

logger = logging.getLogger(__name__)


@triton.jit
def _postrope_stage2_chunked_prefill_fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    threshold_ln_lambda,
    Out,
    Lse,
    MglobalIn,
    MglobalOut,
    Mask,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lsez,
    stride_lseh,
    stride_lsem,
    stride_mgin_z,
    stride_mgin_h,
    stride_mgin_m,
    stride_mgout_z,
    stride_mgout_h,
    stride_mgout_m,
    stride_mask_g0,
    stride_mask_g1,
    stride_mask_b,
    Z,
    H,
    H_KV,
    N_CTX_Q,
    N_CTX_K,
    KV_OFFSET,
    HAS_MGLOBAL_IN: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)

    h_i64 = H
    off_z = off_hz // h_i64
    off_h = off_hz % h_i64
    h_kv_i64 = H_KV
    off_h_kv = off_h // (h_i64 // h_kv_i64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + (offs_m_i64[:, None] * stride_qm + offs_d_i64[None, :] * stride_qk)
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h_kv * stride_kh
        + (offs_n_i64[:, None] * stride_kn + offs_d_i64[None, :] * stride_kk)
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h_kv * stride_vh
        + (offs_n_i64[:, None] * stride_vn + offs_d_i64[None, :] * stride_vk)
    )

    if HAS_MASK:
        mask_ptrs = Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1
    else:
        mask_ptrs = Mask

    q_mask = offs_m[:, None] < N_CTX_Q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    if IS_CAUSAL:
        q_global_pos = KV_OFFSET + offs_m

    if HAS_MGLOBAL_IN:
        mgin_ptrs = (
            MglobalIn
            + off_z * stride_mgin_z
            + off_h * stride_mgin_h
            + offs_m_i64 * stride_mgin_m
        )
        m_global = tl.load(mgin_ptrs, mask=(offs_m < N_CTX_Q), other=-float("inf"))
    else:
        m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    acc_chunk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_chunk = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_chunk = tl.zeros([BLOCK_M], dtype=tl.float32)

    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        do_compute = 1
        if HAS_MASK:
            mask_val = tl.load(mask_ptrs + block_idx * stride_mask_b)
            if mask_val == 0:
                do_compute = 0

        if IS_CAUSAL:
            kv_block_start = KV_OFFSET + start_n
            q_block_max = KV_OFFSET + pid_m * BLOCK_M + BLOCK_M - 1
            if kv_block_start > q_block_max:
                do_compute = 0

        if do_compute == 1:
            start_n_aligned = tl.multiple_of(start_n, BLOCK_N)
            k_mask_1d = (offs_n + start_n_aligned) < N_CTX_K

            k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            valid_mask = q_mask & k_mask_1d[None, :]
            if IS_CAUSAL:
                kv_global_pos = KV_OFFSET + start_n + offs_n
                causal_mask = q_global_pos[:, None] >= kv_global_pos[None, :]
                valid_mask = valid_mask & causal_mask

            qk = tl.where(valid_mask, qk, float("-inf"))
            m_local = tl.max(qk, axis=1)

            diff = m_local - m_global
            max_diff = tl.max(diff, axis=0)
            m_global = tl.maximum(m_global, m_local)

            if max_diff < threshold_ln_lambda:
                do_compute = 0
            else:
                m_chunk_new = tl.maximum(m_chunk, m_local)
                p = tl.math.exp2((qk - m_chunk_new[:, None]) * 1.44269504)
                scale_factor = tl.math.exp2((m_chunk - m_chunk_new) * 1.44269504)
                l_chunk_new = l_chunk * scale_factor + tl.sum(p, axis=1)

                v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0)
                acc_chunk = acc_chunk * scale_factor[:, None]
                acc_chunk = acc_chunk + tl.dot(p.to(v.dtype), v)

                m_chunk = m_chunk_new
                l_chunk = l_chunk_new

        if HAS_MASK:
            tl.store(mask_ptrs + block_idx * stride_mask_b, tl.cast(do_compute, tl.int8))

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        block_idx += 1

    l_chunk_safe = tl.where(l_chunk > 0.0, l_chunk, 1.0)
    acc_chunk = acc_chunk / l_chunk_safe[:, None]
    acc_chunk = tl.where(l_chunk[:, None] > 0.0, acc_chunk, 0.0)

    lse_chunk = tl.where(
        l_chunk > 0.0,
        m_chunk + tl.math.log2(l_chunk_safe) * 0.69314718,
        -float("inf"),
    )

    out_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + (offs_m_i64[:, None] * stride_om + offs_d_i64[None, :] * stride_ok)
    )
    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m_i64 * stride_lsem

    tl.store(out_ptrs, acc_chunk.to(Out.dtype.element_ty), mask=q_mask)
    tl.store(lse_ptrs, lse_chunk, mask=(offs_m < N_CTX_Q))

    mgout_ptrs = (
        MglobalOut
        + off_z * stride_mgout_z
        + off_h * stride_mgout_h
        + offs_m_i64 * stride_mgout_m
    )
    tl.store(mgout_ptrs, m_global, mask=(offs_m < N_CTX_Q))


@triton.jit
def _postrope_stage2_pass1_kernel(
    Q,
    K,
    sm_scale,
    threshold_ln_lambda,
    Lse,
    MglobalIn,
    MglobalOut,
    Mask,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_lsez,
    stride_lseh,
    stride_lsem,
    stride_mgin_z,
    stride_mgin_h,
    stride_mgin_m,
    stride_mgout_z,
    stride_mgout_h,
    stride_mgout_m,
    stride_mask_g0,
    stride_mask_g1,
    stride_mask_b,
    Z,
    H,
    H_KV,
    N_CTX_Q,
    N_CTX_K,
    KV_OFFSET,
    HEAD_DIM,
    HAS_MGLOBAL_IN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)

    h_i64 = H
    off_z = off_hz // h_i64
    off_h = off_hz % h_i64
    h_kv_i64 = H_KV
    off_h_kv = off_h // (h_i64 // h_kv_i64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    q_mask = offs_m[:, None] < N_CTX_Q
    if IS_CAUSAL:
        q_global_pos = KV_OFFSET + offs_m

    if HAS_MGLOBAL_IN:
        mgin_ptrs = (
            MglobalIn
            + off_z * stride_mgin_z
            + off_h * stride_mgin_h
            + offs_m_i64 * stride_mgin_m
        )
        m_global = tl.load(mgin_ptrs, mask=(offs_m < N_CTX_Q), other=-float("inf"))
    else:
        m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    m_chunk = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_chunk = tl.zeros([BLOCK_M], dtype=tl.float32)

    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        do_compute = 1
        if IS_CAUSAL:
            kv_block_start = KV_OFFSET + start_n
            q_block_max = KV_OFFSET + pid_m * BLOCK_M + BLOCK_M - 1
            if kv_block_start > q_block_max:
                do_compute = 0

        if do_compute == 1:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            k_mask_1d = (offs_n + start_n) < N_CTX_K

            for d_start in range(0, HEAD_DIM, BLOCK_DMODEL):
                d_start_i64 = tl.full([1], d_start, dtype=tl.int64)
                q_ptrs = (
                    Q
                    + off_z * stride_qz
                    + off_h * stride_qh
                    + (
                        offs_m_i64[:, None] * stride_qm
                        + (offs_d_i64[None, :] + d_start_i64) * stride_qk
                    )
                )
                k_ptrs = (
                    K
                    + off_z * stride_kz
                    + off_h_kv * stride_kh
                    + (
                        (offs_n_i64[:, None] + start_n) * stride_kn
                        + (offs_d_i64[None, :] + d_start_i64) * stride_kk
                    )
                )

                d_mask = (offs_d + d_start) < HEAD_DIM
                q = tl.load(q_ptrs, mask=q_mask & d_mask[None, :], other=0.0)
                k = tl.load(k_ptrs, mask=k_mask_1d[:, None] & d_mask[None, :], other=0.0)
                qk += tl.dot(q, tl.trans(k))

            qk *= sm_scale
            valid_mask = q_mask & k_mask_1d[None, :]
            if IS_CAUSAL:
                kv_global_pos = KV_OFFSET + start_n + offs_n
                causal_mask = q_global_pos[:, None] >= kv_global_pos[None, :]
                valid_mask = valid_mask & causal_mask

            qk = tl.where(valid_mask, qk, float("-inf"))
            m_local = tl.max(qk, axis=1)

            diff = m_local - m_global
            max_diff = tl.max(diff, axis=0)
            m_global = tl.maximum(m_global, m_local)

            if max_diff < threshold_ln_lambda:
                do_compute = 0
            else:
                m_chunk_new = tl.maximum(m_chunk, m_local)
                p = tl.math.exp2((qk - m_chunk_new[:, None]) * 1.44269504)
                scale_factor = tl.math.exp2((m_chunk - m_chunk_new) * 1.44269504)
                l_chunk = l_chunk * scale_factor + tl.sum(p, axis=1)
                m_chunk = m_chunk_new

        tl.store(
            Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1 + block_idx * stride_mask_b,
            tl.cast(do_compute, tl.int8),
        )
        block_idx += 1

    l_chunk_safe = tl.where(l_chunk > 0.0, l_chunk, 1.0)
    lse_chunk = tl.where(
        l_chunk > 0.0,
        m_chunk + tl.math.log2(l_chunk_safe) * 0.69314718,
        -float("inf"),
    )

    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m_i64 * stride_lsem
    tl.store(lse_ptrs, lse_chunk, mask=(offs_m < N_CTX_Q))

    mgout_ptrs = (
        MglobalOut
        + off_z * stride_mgout_z
        + off_h * stride_mgout_h
        + offs_m_i64 * stride_mgout_m
    )
    tl.store(mgout_ptrs, m_global, mask=(offs_m < N_CTX_Q))


@triton.jit
def _postrope_stage2_pass2_kernel(
    Q,
    K,
    V,
    Lse,
    Mask,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_lsez,
    stride_lseh,
    stride_lsem,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_mask_g0,
    stride_mask_g1,
    stride_mask_b,
    Z,
    H,
    H_KV,
    N_CTX_Q,
    N_CTX_K,
    KV_OFFSET,
    HEAD_DIM,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    pid_do = tl.program_id(2).to(tl.int64)

    h_i64 = H
    off_z = off_hz // h_i64
    off_h = off_hz % h_i64
    h_kv_i64 = H_KV
    off_h_kv = off_h // (h_i64 // h_kv_i64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_do = pid_do * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)
    offs_do_i64 = offs_do.to(tl.int64)

    q_mask = offs_m[:, None] < N_CTX_Q
    if IS_CAUSAL:
        q_global_pos = KV_OFFSET + offs_m

    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m_i64 * stride_lsem
    lse_row = tl.load(lse_ptrs, mask=(offs_m < N_CTX_Q), other=-float("inf"))
    acc = tl.zeros([BLOCK_M, BLOCK_OUT], dtype=tl.float32)

    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        do_compute = tl.load(
            Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1 + block_idx * stride_mask_b
        )
        if do_compute != 0:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            k_mask_1d = (offs_n + start_n) < N_CTX_K

            for d_start in range(0, HEAD_DIM, BLOCK_DMODEL):
                d_start_i64 = tl.full([1], d_start, dtype=tl.int64)
                q_ptrs = (
                    Q
                    + off_z * stride_qz
                    + off_h * stride_qh
                    + (
                        offs_m_i64[:, None] * stride_qm
                        + (offs_d_i64[None, :] + d_start_i64) * stride_qk
                    )
                )
                k_ptrs = (
                    K
                    + off_z * stride_kz
                    + off_h_kv * stride_kh
                    + (
                        (offs_n_i64[:, None] + start_n) * stride_kn
                        + (offs_d_i64[None, :] + d_start_i64) * stride_kk
                    )
                )

                d_mask = (offs_d + d_start) < HEAD_DIM
                q = tl.load(q_ptrs, mask=q_mask & d_mask[None, :], other=0.0)
                k = tl.load(k_ptrs, mask=k_mask_1d[:, None] & d_mask[None, :], other=0.0)
                qk += tl.dot(q, tl.trans(k))

            qk *= sm_scale
            valid_mask = q_mask & k_mask_1d[None, :]
            if IS_CAUSAL:
                kv_global_pos = KV_OFFSET + start_n + offs_n
                causal_mask = q_global_pos[:, None] >= kv_global_pos[None, :]
                valid_mask = valid_mask & causal_mask

            qk = tl.where(valid_mask, qk, float("-inf"))
            p = tl.math.exp2((qk - lse_row[:, None]) * 1.44269504)

            v_ptrs = (
                V
                + off_z * stride_vz
                + off_h_kv * stride_vh
                + (
                    (offs_n_i64[:, None] + start_n) * stride_vn
                    + offs_do_i64[None, :] * stride_vk
                )
            )
            do_mask = offs_do < HEAD_DIM
            v = tl.load(v_ptrs, mask=k_mask_1d[:, None] & do_mask[None, :], other=0.0)
            acc += tl.dot(p.to(v.dtype), v)

        block_idx += 1

    out_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + (offs_m_i64[:, None] * stride_om + offs_do_i64[None, :] * stride_ok)
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask & (offs_do[None, :] < HEAD_DIM))


def _postrope_flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.flash_attn_interface import flash_attn_func

    _, seqlen_q, _, headdim = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    out, lse, _ = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    lse = lse[:, :, :seqlen_q]
    return out, lse


def _postrope_merge_attention_outputs(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse1_f = lse1.float()
    lse2_f = lse2.float()
    max_lse = torch.maximum(lse1_f, lse2_f)
    exp1 = torch.exp(lse1_f - max_lse)
    exp2 = torch.exp(lse2_f - max_lse)
    denom = (exp1 + exp2).clamp_min_(1e-20)
    lse_merged = max_lse + torch.log(denom)

    w1 = (exp1 / denom).transpose(1, 2).unsqueeze(-1)
    w2 = (exp2 / denom).transpose(1, 2).unsqueeze(-1)
    o_merged = o1 * w1.to(dtype=o1.dtype) + o2 * w2.to(dtype=o2.dtype)
    return o_merged, lse_merged.to(dtype=lse1.dtype)


def _postrope_merge_attention_outputs_inplace(
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    o_new: torch.Tensor,
    lse_new: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_acc_f = lse_acc.float()
    lse_new_f = lse_new.float()
    max_lse = torch.maximum(lse_acc_f, lse_new_f)
    exp1 = torch.exp(lse_acc_f - max_lse)
    exp2 = torch.exp(lse_new_f - max_lse)
    denom = (exp1 + exp2).clamp_min_(1e-20)
    lse_merged = max_lse + torch.log(denom)

    w1 = (exp1 / denom).transpose(1, 2).unsqueeze(-1).to(dtype=o_acc.dtype)
    w2 = (exp2 / denom).transpose(1, 2).unsqueeze(-1).to(dtype=o_acc.dtype)
    o_acc.mul_(w1)
    o_acc.add_(o_new * w2)
    lse_acc.copy_(lse_merged.to(dtype=lse_acc.dtype))
    return o_acc, lse_acc

def _postrope_stage2_chunked_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    threshold_ln_lambda: float = -6.9,
    m_global_in: Optional[torch.Tensor] = None,
    mask_buffer: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    kv_offset: int = 0,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    batch, q_len, num_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    two_pass = os.environ.get("POSTROPE_STAGE2_TWO_PASS", "0") == "1"
    out = torch.empty_like(q)
    lse = torch.empty((batch, num_heads, q_len), device=q.device, dtype=torch.float32)
    m_global_out = torch.empty((batch, num_heads, q_len), device=q.device, dtype=torch.float32)

    block_m = int(os.environ.get("POSTROPE_STAGE2_BLOCK_M", "128"))
    block_n = int(os.environ.get("POSTROPE_STAGE2_BLOCK_N", "64"))
    num_warps = int(os.environ.get("POSTROPE_STAGE2_NUM_WARPS", "8"))
    num_stages = int(os.environ.get("POSTROPE_STAGE2_NUM_STAGES", "2"))
    maxnreg_env = os.environ.get("POSTROPE_STAGE2_MAXNREG")
    maxnreg = int(maxnreg_env) if maxnreg_env else None
    grid = (triton.cdiv(q_len, block_m), batch * num_heads)
    sm_scale = 1.0 / (head_dim**0.5)

    has_mglobal_in = m_global_in is not None
    mgin = m_global_in if has_mglobal_in else lse
    mgin_strides = mgin.stride()

    has_mask = mask_buffer is not None
    mask_tensor = mask_buffer if has_mask else lse
    mask_strides = mask_tensor.stride()

    if two_pass:
        if not has_mask:
            num_sub = triton.cdiv(kv_len, block_n)
            mask_tensor = torch.empty(
                (grid[0], grid[1], num_sub), device=q.device, dtype=torch.int8
            )
            mask_strides = mask_tensor.stride()

        _postrope_stage2_pass1_kernel[grid](
            q,
            k,
            sm_scale,
            threshold_ln_lambda,
            lse,
            mgin,
            m_global_out,
            mask_tensor,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            mgin_strides[0],
            mgin_strides[1],
            mgin_strides[2],
            m_global_out.stride(0),
            m_global_out.stride(1),
            m_global_out.stride(2),
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            kv_offset,
            head_dim,
            has_mglobal_in,
            is_causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_DMODEL=64,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )

        grid2 = (grid[0], grid[1], triton.cdiv(head_dim, 64))
        _postrope_stage2_pass2_kernel[grid2](
            q,
            k,
            v,
            lse,
            mask_tensor,
            out,
            sm_scale,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            kv_offset,
            head_dim,
            is_causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_DMODEL=64,
            BLOCK_OUT=64,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
    else:
        _postrope_stage2_chunked_prefill_fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            threshold_ln_lambda,
            out,
            lse,
            mgin,
            m_global_out,
            mask_tensor,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            mgin_strides[0],
            mgin_strides[1],
            mgin_strides[2],
            m_global_out.stride(0),
            m_global_out.stride(1),
            m_global_out.stride(2),
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            batch,
            num_heads,
            num_kv_heads,
            q_len,
            kv_len,
            kv_offset,
            has_mglobal_in,
            has_mask,
            is_causal,
            BLOCK_M=block_m,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )

    return out, lse, m_global_out


class PostRoPEPolicy(SparsePolicy):
    """Explicit post-RoPE policy with a self-contained sparse prefill path."""

    supports_prefill = True
    supports_decode = True
    apply_rope_in_attention = False

    # Stage-1 pooled selection granularity. We keep fine 128-token summaries for
    # estimation, while the actual offload / IO chunk remains the normal 4096-token
    # KV cache block selected by the runtime.
    ESTIMATE_CHUNK_SIZE = 128

    # Conservative defaults: prioritize correctness over aggressiveness while
    # still allowing stage-1 parent-chunk pruning.
    TOP_P = 0.90
    THETA = 0.60
    SMALL_BLOCK_THETA = 0.45
    FORCE_RECENT_BLOCKS = 1
    RECENT_TOKENS_BUDGET = 4096
    FIXED_LAMBDA = 1e-4
    STAGE2_DENSITY_LOG_LAYER = 5

    def __init__(self):
        self._num_heads = 0
        self._num_kv_heads = 0
        self._head_dim = 0
        self._heads_per_group = 1
        self._num_layers = 0
        self._kvcache_block_size = 4096

        # layer_id -> {cpu_block_id -> (pooled_k_chunk [H_kv, D], self_cos_chunk [H_kv])}
        self._k_summary_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Stage-1 selection stats
        self._stats_total_available_blocks = 0
        self._stats_total_selected_blocks = 0
        self._stats_total_persistent_blocks = 0
        self._stats_num_chunks = 0
        self._stats_cache_miss_fallbacks = 0
        self._last_selection_info: Dict[str, float | int] = {}

        # Stage-2 sparse-compute stats
        self._stats_stage2_num_chunks = 0
        self._stats_stage2_skipped_subblocks = 0
        self._stats_stage2_total_subblocks = 0

    # ---------------------------------------------------------------------
    # Metadata / lifecycle
    # ---------------------------------------------------------------------

    def initialize(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: torch.device = None,
    ) -> None:
        del num_cpu_blocks, dtype, device
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._k_summary_cache = {layer_id: {} for layer_id in range(num_layers)}

    def alloc_policy_metadata(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_cpu_offload: bool = False,
        num_layers: int = 64,
    ) -> None:
        del max_seq_len, dtype, device
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._heads_per_group = max(1, num_heads // max(1, num_kv_heads))
        if enable_cpu_offload and (not self._k_summary_cache or self._num_layers != num_layers):
            self._num_layers = num_layers
            self._k_summary_cache = {layer_id: {} for layer_id in range(num_layers)}

    def reset_request_state(self) -> None:
        for layer_cache in self._k_summary_cache.values():
            layer_cache.clear()
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats_total_available_blocks = 0
        self._stats_total_selected_blocks = 0
        self._stats_total_persistent_blocks = 0
        self._stats_num_chunks = 0
        self._stats_cache_miss_fallbacks = 0
        self._last_selection_info = {}
        self._stats_stage2_num_chunks = 0
        self._stats_stage2_skipped_subblocks = 0
        self._stats_stage2_total_subblocks = 0

    def get_density_stats(self) -> dict:
        density = 1.0
        persistent_density = 0.0
        if self._stats_total_available_blocks > 0:
            density = self._stats_total_selected_blocks / self._stats_total_available_blocks
            persistent_density = (
                self._stats_total_persistent_blocks / self._stats_total_available_blocks
            )

        stage2_skip_rate = 0.0
        if self._stats_stage2_total_subblocks > 0:
            stage2_skip_rate = (
                self._stats_stage2_skipped_subblocks / self._stats_stage2_total_subblocks
            )

        return {
            "total_available_blocks": self._stats_total_available_blocks,
            "total_selected_blocks": self._stats_total_selected_blocks,
            "total_persistent_blocks": self._stats_total_persistent_blocks,
            "num_chunks": self._stats_num_chunks,
            "overall_density": density,
            "persistent_density": persistent_density,
            "cache_miss_fallbacks": self._stats_cache_miss_fallbacks,
            "stage2_num_chunks": self._stats_stage2_num_chunks,
            "stage2_total_subblocks": self._stats_stage2_total_subblocks,
            "stage2_skipped_subblocks": self._stats_stage2_skipped_subblocks,
            "stage2_skip_rate": stage2_skip_rate,
        }

    def print_density_stats(self) -> None:
        stats = self.get_density_stats()
        logger.info(
            "[PostRoPE Sparge] Density Stats: chunks=%s, selected=%s/%s (%.1f%%), "
            "persistent=%s/%s (%.1f%%), cache_miss_fallbacks=%s, stage2_skip_rate=%.2f%%",
            stats["num_chunks"],
            stats["total_selected_blocks"],
            stats["total_available_blocks"],
            stats["overall_density"] * 100.0,
            stats["total_persistent_blocks"],
            stats["total_available_blocks"],
            stats["persistent_density"] * 100.0,
            stats["cache_miss_fallbacks"],
            stats["stage2_skip_rate"] * 100.0,
        )

    # ---------------------------------------------------------------------
    # Stage-1 pooled estimation helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _mean_pool(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Mean-pool `[T, H, D]` into `[G, H, D]` with partial-tail support."""
        if x.numel() == 0:
            return x.new_empty((0, x.shape[1], x.shape[2]), dtype=torch.float32)
        full_groups, tail = divmod(x.shape[0], chunk_size)
        x_float = x.float()
        if tail == 0:
            return x_float.view(full_groups, chunk_size, x.shape[1], x.shape[2]).mean(dim=1)

        groups = []
        if full_groups > 0:
            groups.append(
                x_float[: full_groups * chunk_size]
                .view(full_groups, chunk_size, x.shape[1], x.shape[2])
                .mean(dim=1)
            )
        groups.append(x_float[full_groups * chunk_size :].mean(dim=0, keepdim=True))
        return torch.cat(groups, dim=0)

    @staticmethod
    def _compute_self_cosine(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Compute SpargeAttn-style self-cosine per pooled group and head."""
        if x.numel() == 0:
            return x.new_empty((0, x.shape[1]), dtype=torch.float32)
        # Mean pairwise cosine equals ||mean(normalized_tokens)||^2, which is
        # mathematically identical to averaging the full Gram matrix while
        # avoiding the explicit O(T^2) construction.
        def reduce_groups(part: torch.Tensor) -> torch.Tensor:
            part = part.permute(0, 2, 1, 3).contiguous()  # [G, H, T, D]
            norm = part.norm(dim=-1, keepdim=True).clamp_min_(1e-6)
            normalized = part / norm
            mean_vec = normalized.mean(dim=2)
            return (mean_vec * mean_vec).sum(dim=-1)

        full_groups, tail = divmod(x.shape[0], chunk_size)
        x_float = x.float()
        values = []
        if full_groups > 0:
            full = x_float[: full_groups * chunk_size].view(
                full_groups, chunk_size, x.shape[1], x.shape[2]
            )
            values.append(reduce_groups(full))
        if tail > 0:
            values.append(reduce_groups(x_float[full_groups * chunk_size :].unsqueeze(0)))
        return torch.cat(values, dim=0)

    def _fold_q_heads(self, pooled_q: torch.Tensor) -> torch.Tensor:
        """Map query heads `[G, H_q, D]` to KV-head groups `[G, H_kv, D]`."""
        if pooled_q.shape[1] == self._num_kv_heads:
            return pooled_q
        return pooled_q.reshape(
            pooled_q.shape[0],
            self._num_kv_heads,
            self._heads_per_group,
            pooled_q.shape[-1],
        ).mean(dim=2)

    def _fold_q_head_mask_any(self, q_mask: torch.Tensor) -> torch.Tensor:
        """Map query-head boolean masks `[G, H_q]` to KV-head groups `[G, H_kv]` via OR."""
        if q_mask.shape[1] == self._num_kv_heads:
            return q_mask
        return q_mask.reshape(
            q_mask.shape[0],
            self._num_kv_heads,
            self._heads_per_group,
        ).any(dim=2)

    def _record_selection_info(
        self,
        *,
        available_blocks: int,
        selected_blocks: int,
        persistent_blocks: int,
        top_p_selected_blocks: int,
        q_persistent_groups: int = 0,
        q_force_heads: int = 0,
        q_force_keep_all: bool = False,
    ) -> None:
        self._last_selection_info = {
            "available_blocks": available_blocks,
            "selected_blocks": selected_blocks,
            "persistent_blocks": persistent_blocks,
            "top_p_selected_blocks": top_p_selected_blocks,
            "selected_density": selected_blocks / max(1, available_blocks),
            "persistent_density": persistent_blocks / max(1, available_blocks),
            "q_persistent_groups": q_persistent_groups,
            "q_force_heads": q_force_heads,
            "q_force_keep_all": q_force_keep_all,
        }

    def _get_force_recent_blocks(self) -> int:
        block_size = max(1, int(self._kvcache_block_size))
        return max(
            self.FORCE_RECENT_BLOCKS,
            (self.RECENT_TOKENS_BUDGET + block_size - 1) // block_size,
        )

    def _get_stage1_theta(self) -> float:
        if self._kvcache_block_size <= 1024:
            return self.SMALL_BLOCK_THETA
        return self.THETA

    def _get_lambda(self, seq_len: int) -> float:
        return self.FIXED_LAMBDA

    def _record_stage2_mask(self, mask_buffer: torch.Tensor) -> None:
        total_subblocks = int(mask_buffer.numel())
        kept_subblocks = int(mask_buffer.sum().item())
        self._stats_stage2_total_subblocks += total_subblocks
        self._stats_stage2_skipped_subblocks += total_subblocks - kept_subblocks

    def _select_historical_blocks_gpu(
        self,
        layer_id: int,
        available_blocks: List[int],
        q: torch.Tensor,
        record_info: bool,
    ) -> List[int]:
        """Run stage-1 chunk selection fully on GPU.

        The offload IO granularity is a parent KV chunk, so stage 1 returns a
        1D selected-block list rather than the paper's full 2D `M_g[i, j]`
        mask. Within that constraint we keep both paper-style protection rules:

        1. low-self-similarity K chunks are forced on and excluded from top-p;
        2. low-self-similarity Q groups are treated as fix rows. Since the
           current runtime only accepts a 1D selected-block list for the whole
           chunk, any KV-head group with at least one such Q group forces all
           historical blocks on for that head.
        """
        layer_cache = self._k_summary_cache.get(layer_id, {})
        missing = [bid for bid in available_blocks if bid not in layer_cache]
        if missing:
            self._stats_cache_miss_fallbacks += 1
            if record_info:
                self._record_selection_info(
                    available_blocks=len(available_blocks),
                    selected_blocks=len(available_blocks),
                    persistent_blocks=0,
                    top_p_selected_blocks=len(available_blocks),
                )
            logger.warning(
                "[PostRoPE Sparge] summary cache miss on layer=%s blocks=%s -> fallback to full",
                layer_id,
                missing[:4],
            )
            return available_blocks

        q_pooled = self._fold_q_heads(self._mean_pool(q, self.ESTIMATE_CHUNK_SIZE)).to(torch.float32)
        if q_pooled.numel() == 0:
            if record_info:
                self._record_selection_info(
                    available_blocks=len(available_blocks),
                    selected_blocks=len(available_blocks),
                    persistent_blocks=0,
                    top_p_selected_blocks=len(available_blocks),
                )
            return available_blocks

        theta = self._get_stage1_theta()
        q_low_sim_heads = self._compute_self_cosine(q, self.ESTIMATE_CHUNK_SIZE) < theta  # [G_q, H_q]
        q_persistent_groups = self._fold_q_head_mask_any(q_low_sim_heads) if q_low_sim_heads.numel() > 0 else torch.zeros(
            (0, q_pooled.shape[1]), device=q.device, dtype=torch.bool
        )
        q_force_heads = q_persistent_groups.any(dim=0) if q_persistent_groups.numel() > 0 else torch.zeros(
            (q_pooled.shape[1],), device=q.device, dtype=torch.bool
        )

        device = q.device
        pooled_k_chunks = []
        chunk_self_cos = []
        for bid in available_blocks:
            pooled_k, self_cos_k = layer_cache[bid]
            pooled_k_chunks.append(pooled_k.to(device=device, dtype=torch.float32))
            chunk_self_cos.append(self_cos_k.to(device=device, dtype=torch.float32))

        k_chunk = torch.stack(pooled_k_chunks, dim=0)  # [B, H_kv, D]
        k_chunk_self_cos = torch.stack(chunk_self_cos, dim=0)  # [B, H_kv]
        num_blocks = k_chunk.shape[0]
        h_kv = q_pooled.shape[1]

        q_t = q_pooled.permute(1, 0, 2).contiguous()  # [H_kv, G_q, D]
        k_t = k_chunk.permute(1, 2, 0).contiguous()  # [H_kv, D, B]
        scores = torch.bmm(q_t, k_t) * (self._head_dim ** -0.5)  # [H_kv, G_q, B]

        persistent_chunks = (k_chunk_self_cos < theta).any(dim=-1)  # [B]
        masked_scores = scores.masked_fill(
            persistent_chunks.view(1, 1, num_blocks),
            float("-inf"),
        )

        if bool((~persistent_chunks).any()):
            probs = torch.softmax(masked_scores, dim=-1)
            avg_probs = probs.mean(dim=1)  # [H_kv, B]
        else:
            avg_probs = torch.zeros((h_kv, num_blocks), device=device, dtype=scores.dtype)

        selected_by_head = torch.zeros((h_kv, num_blocks), dtype=torch.bool, device=device)
        if persistent_chunks.any():
            selected_by_head |= persistent_chunks.view(1, num_blocks)
        if bool(q_force_heads.any()):
            selected_by_head[q_force_heads] = True

        selectable = ~persistent_chunks
        selectable_heads = ~q_force_heads
        if bool(selectable.any()) and bool(selectable_heads.any()):
            active_head_idx = selectable_heads.nonzero(as_tuple=True)[0]
            probs_for_sort = avg_probs[active_head_idx][:, selectable]
            sorted_probs, sorted_local_idx = torch.sort(
                probs_for_sort, dim=-1, descending=True
            )
            cdf = torch.cumsum(sorted_probs, dim=-1)
            cdf_flat = cdf.reshape(-1, probs_for_sort.shape[-1])
            target = torch.full(
                (cdf_flat.shape[0], 1),
                self.TOP_P,
                device=device,
                dtype=cdf_flat.dtype,
            )
            num_to_select = torch.searchsorted(cdf_flat, target, right=True).squeeze(-1)
            num_to_select = (num_to_select + 1).clamp(
                min=1, max=probs_for_sort.shape[-1]
            )
            selectable_idx = selectable.nonzero(as_tuple=True)[0]
            keep_mask_sorted = (
                torch.arange(
                    probs_for_sort.shape[-1], device=device, dtype=num_to_select.dtype
                ).unsqueeze(0)
                < num_to_select.unsqueeze(1)
            )
            selected_local = torch.zeros_like(probs_for_sort, dtype=torch.bool)
            selected_local.scatter_(1, sorted_local_idx, keep_mask_sorted)

            active_selected = selected_by_head.index_select(0, active_head_idx)
            active_selected[:, selectable_idx] |= selected_local
            selected_by_head.index_copy_(0, active_head_idx, active_selected)

        if num_blocks > 0:
            selected_by_head[:, 0] = True  # attention sink / anchor chunk

        selected_chunks = selected_by_head.any(dim=0)
        force_recent_blocks = self._get_force_recent_blocks()
        if force_recent_blocks > 0:
            selected_chunks[-force_recent_blocks:] = True

        if not bool(selected_chunks.any()):
            selected_chunks[-1] = True

        selected_blocks = [
            bid for bid, keep in zip(available_blocks, selected_chunks.tolist()) if keep
        ]
        if record_info:
            persistent_count = int(persistent_chunks.sum().item())
            top_p_selected_count = max(0, len(selected_blocks) - persistent_count)
            self._record_selection_info(
                available_blocks=len(available_blocks),
                selected_blocks=len(selected_blocks),
                persistent_blocks=persistent_count,
                top_p_selected_blocks=top_p_selected_count,
                q_persistent_groups=int(q_persistent_groups.sum().item()),
                q_force_heads=int(q_force_heads.sum().item()),
                q_force_keep_all=bool(q_force_heads.any().item()),
            )
        return selected_blocks

    # ---------------------------------------------------------------------
    # Selection / compute
    # ---------------------------------------------------------------------

    def select_blocks(
        self,
        available_blocks: List[int],
        offload_engine: "OffloadEngine",
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> List[int]:
        del offload_engine, k

        if ctx.layer_id == 0 and available_blocks:
            self._stats_total_available_blocks += len(available_blocks)
            self._stats_num_chunks += 1

        # Keep decode conservative for now.
        if not ctx.is_prefill or not available_blocks:
            selected = available_blocks
        else:
            selected = self._select_historical_blocks_gpu(
                ctx.layer_id,
                available_blocks,
                q,
                record_info=(ctx.layer_id == 0),
            )

        if ctx.layer_id == 0 and os.environ.get("POSTROPE_LOG_SELECTION", "0") == "1":
            info = self._last_selection_info or {
                "selected_blocks": len(selected),
                "persistent_blocks": 0,
                "top_p_selected_blocks": len(selected),
                "selected_density": len(selected) / max(1, len(available_blocks)),
                "persistent_density": 0.0,
                "q_persistent_groups": 0,
                "q_force_heads": 0,
                "q_force_keep_all": False,
            }
            self._stats_total_selected_blocks += int(info["selected_blocks"])
            self._stats_total_persistent_blocks += int(info["persistent_blocks"])
            selected_count = int(info["selected_blocks"])
            persistent_count = int(info["persistent_blocks"])
            available_count = max(1, len(available_blocks))
            logger.info(
                "[PostRoPE Sparge] chunk=%s select_blocks: %s -> %s blocks (%.1f%%), "
                "persistent=%s (%.1f%%), top_p_selected=%s, q_persistent_groups=%s, q_force_heads=%s, q_force_keep_all=%s",
                ctx.query_chunk_idx,
                len(available_blocks),
                selected_count,
                100.0 * selected_count / available_count,
                persistent_count,
                100.0 * persistent_count / available_count,
                int(info["top_p_selected_blocks"]),
                int(info.get("q_persistent_groups", 0)),
                int(info.get("q_force_heads", 0)),
                int(bool(info.get("q_force_keep_all", False))),
            )

        return selected

    def compute_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        layer_id: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        del layer_id
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=block_tables,
        )

    def compute_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        layer_id: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_with_kvcache

        del layer_id
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )

    def compute_chunked_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        current_chunk_idx: int,
        seq: "Sequence",
        num_tokens: int,
        selected_blocks: List[int],
    ) -> torch.Tensor:
        del k, v

        total_seq_len = len(seq) if seq else num_tokens
        lambda_val = self._get_lambda(total_seq_len)
        ln_lambda = math.log(lambda_val)

        if layer_id == 0:
            self._stats_stage2_num_chunks += 1
            if os.environ.get("POSTROPE_LOG_STAGE2_CHUNKS", "0") == "1":
                logger.info(
                    "[PostRoPE Sparge][Stage2] Chunk %s: seq_len=%s, lambda=%.6f, ln(lambda)=%.4f",
                    current_chunk_idx,
                    total_seq_len,
                    lambda_val,
                    ln_lambda,
                )

        q_len = q.shape[0]
        num_heads = q.shape[1]
        compute_stream = offload_engine.compute_stream
        q_input = q.unsqueeze(0)
        historical_o = None
        historical_lse = None
        historical_m_global = None

        collect_density = (
            os.environ.get("POSTROPE_LOG_STAGE2_DENSITY", "0") == "1"
            and layer_id == self.STAGE2_DENSITY_LOG_LAYER
        )
        collect_stage2_stats = os.environ.get("POSTROPE_COLLECT_STAGE2_STATS", "0") == "1"
        compute_density_sum = 0.0
        per_head_kv_density_list = []
        per_head_compute_density_list = []
        num_density_measurements = 0

        TRITON_BLOCK_M = int(os.environ.get("POSTROPE_STAGE2_BLOCK_M", "128"))
        TRITON_BLOCK_N = int(os.environ.get("POSTROPE_STAGE2_BLOCK_N", "64"))
        grid_0 = (q_len + TRITON_BLOCK_M - 1) // TRITON_BLOCK_M
        grid_1 = num_heads
        num_kv_subblocks = kvcache_manager.block_size // TRITON_BLOCK_N

        def get_mask_buffer(is_causal: bool = False, kv_len_override: Optional[int] = None):
            num_sub = (
                num_kv_subblocks
                if kv_len_override is None
                else (kv_len_override + TRITON_BLOCK_N - 1) // TRITON_BLOCK_N
            )
            mask = torch.ones(
                (grid_0, grid_1, num_sub), device=q.device, dtype=torch.int8
            )

            if is_causal:
                q_end = (
                    (torch.arange(grid_0, device=q.device, dtype=torch.int32) + 1)
                    * TRITON_BLOCK_M
                )
                kv_start = (
                    torch.arange(num_sub, device=q.device, dtype=torch.int32)
                    * TRITON_BLOCK_N
                )
                mask &= (kv_start.unsqueeze(0) < q_end.unsqueeze(1)).unsqueeze(1)
            return mask

        def collect_per_head_kv_density(mask_buf: torch.Tensor) -> torch.Tensor:
            required_kv = mask_buf.any(dim=0).float()
            return required_kv.mean(dim=-1)

        def collect_per_head_compute_density(mask_buf: torch.Tensor) -> torch.Tensor:
            return mask_buf.float().mean(dim=(0, 2))

        cpu_block_table = selected_blocks
        if cpu_block_table:
            load_slots = list(range(offload_engine.num_ring_slots))
            num_slots = len(load_slots)
            num_blocks = len(cpu_block_table)
            group_size = min(
                num_slots,
                max(1, int(os.environ.get("POSTROPE_STAGE2_GROUP_BLOCKS", str(num_slots)))),
            )

            if num_slots > 0:
                for group_start in range(0, num_blocks, group_size):
                    group_blocks = cpu_block_table[group_start : group_start + group_size]
                    group_slots = load_slots[: len(group_blocks)]

                    for slot_idx, cpu_block_id in zip(group_slots, group_blocks):
                        offload_engine.load_to_slot_layer(
                            slot_idx,
                            layer_id,
                            cpu_block_id,
                            chunk_idx=cpu_block_id,
                        )

                    for slot_idx in group_slots:
                        offload_engine.wait_slot_layer(slot_idx)

                    with torch.cuda.stream(compute_stream):
                        if getattr(offload_engine, "is_head_first", False):
                            k_group = []
                            v_group = []
                            for slot_idx in group_slots:
                                prev_k, prev_v = offload_engine.get_kv_for_slot(slot_idx)
                                k_group.append(prev_k)
                                v_group.append(prev_v)
                            k_input = (
                                k_group[0]
                                if len(k_group) == 1
                                else torch.cat(k_group, dim=1)
                            )
                            v_input = (
                                v_group[0]
                                if len(v_group) == 1
                                else torch.cat(v_group, dim=1)
                            )
                        else:
                            k_input = offload_engine.k_cache_gpu[: len(group_slots)].reshape(
                                1,
                                len(group_slots) * kvcache_manager.block_size,
                                offload_engine.num_kv_heads,
                                offload_engine.head_dim,
                            )
                            v_input = offload_engine.v_cache_gpu[: len(group_slots)].reshape(
                                1,
                                len(group_slots) * kvcache_manager.block_size,
                                offload_engine.num_kv_heads,
                                offload_engine.head_dim,
                            )

                        mask_buffer = (
                            get_mask_buffer(
                                is_causal=False, kv_len_override=k_input.shape[1]
                            )
                            if (collect_stage2_stats or collect_density)
                            else None
                        )
                        out, lse, m_global_out = _postrope_stage2_chunked_prefill(
                            q=q_input,
                            k=k_input,
                            v=v_input,
                            threshold_ln_lambda=ln_lambda,
                            m_global_in=historical_m_global,
                            mask_buffer=mask_buffer,
                        )
                        if collect_stage2_stats:
                            self._record_stage2_mask(mask_buffer)

                        historical_m_global = m_global_out

                        if collect_density:
                            compute_stream.synchronize()
                            compute_density_sum += mask_buffer.float().mean().item()
                            per_head_kv_density_list.append(
                                collect_per_head_kv_density(mask_buffer).cpu()
                            )
                            per_head_compute_density_list.append(
                                collect_per_head_compute_density(mask_buffer).cpu()
                            )
                            num_density_measurements += 1

                        for slot_idx in group_slots:
                            offload_engine.record_slot_compute_done(slot_idx)

                        block_o = out
                        block_lse = lse
                        if historical_o is None:
                            historical_o, historical_lse = block_o, block_lse
                        else:
                            historical_o, historical_lse = _postrope_merge_attention_outputs(
                                historical_o, historical_lse, block_o, block_lse
                            )

        with torch.cuda.stream(compute_stream):
            k_curr, v_curr = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)
            k_curr_input = k_curr
            v_curr_input = v_curr

            kv_offset = len(selected_blocks) * kvcache_manager.block_size
            curr_mask_buffer = (
                get_mask_buffer(is_causal=False, kv_len_override=num_tokens)
                if (collect_stage2_stats or collect_density)
                else None
            )
            out_curr, lse_curr, _ = _postrope_stage2_chunked_prefill(
                q=q_input,
                k=k_curr_input,
                v=v_curr_input,
                threshold_ln_lambda=ln_lambda,
                m_global_in=historical_m_global,
                mask_buffer=curr_mask_buffer,
                is_causal=True,
                kv_offset=kv_offset,
            )
            if collect_stage2_stats:
                self._record_stage2_mask(curr_mask_buffer)

            if collect_density:
                compute_stream.synchronize()
                compute_density_sum += curr_mask_buffer.float().mean().item()
                per_head_kv_density_list.append(
                    collect_per_head_kv_density(curr_mask_buffer).cpu()
                )
                per_head_compute_density_list.append(
                    collect_per_head_compute_density(curr_mask_buffer).cpu()
                )
                num_density_measurements += 1

            block_o = out_curr
            block_lse = lse_curr
            if historical_o is None:
                final_o = block_o
            else:
                final_o, _ = _postrope_merge_attention_outputs(
                    historical_o, historical_lse, block_o, block_lse
                )

        if num_density_measurements > 0:
            avg_comp = compute_density_sum / num_density_measurements
            per_head_kv_avg = torch.stack(per_head_kv_density_list).mean(dim=0)
            per_head_comp_avg = torch.stack(per_head_compute_density_list).mean(dim=0)
            avg_req = per_head_kv_avg.mean().item()
            kv_head_details = ", ".join(
                f"H{i}={per_head_kv_avg[i].item() * 100:.1f}%"
                for i in range(len(per_head_kv_avg))
            )
            comp_head_details = ", ".join(
                f"H{i}={per_head_comp_avg[i].item() * 100:.1f}%"
                for i in range(len(per_head_comp_avg))
            )
            logger.info(
                "[PostRoPE Sparge][Stage2] Layer %s Chunk %s Stats: Compute Density=%.2f%%, Required KV Density(avg)=%.2f%%",
                layer_id,
                current_chunk_idx,
                avg_comp * 100.0,
                avg_req * 100.0,
            )
            logger.info(
                "[PostRoPE Sparge][Stage2] Layer %s Chunk %s Per-Head KV Density: %s",
                layer_id,
                current_chunk_idx,
                kv_head_details,
            )
            logger.info(
                "[PostRoPE Sparge][Stage2] Layer %s Chunk %s Per-Head Compute Density: %s",
                layer_id,
                current_chunk_idx,
                comp_head_details,
            )

        torch.cuda.default_stream().wait_stream(compute_stream)
        return final_o.squeeze(0)

    def compute_chunked_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        softmax_scale: float,
        offload_engine: "OffloadEngine",
        kvcache_manager: "KVCacheManager",
        seq: "Sequence",
        selected_blocks: List[int],
    ) -> torch.Tensor:
        q_batched = q.unsqueeze(1)
        cpu_block_table = selected_blocks
        if layer_id == 0:
            logger.debug(
                "[PostRoPE Sparge][Decode] selected_blocks=%s, seq.block_table=%s",
                len(selected_blocks),
                list(seq.block_table),
            )
        if not cpu_block_table:
            raise RuntimeError(
                "Chunked decode attention failed: no prefilled CPU blocks available"
            )

        block_size = kvcache_manager.block_size
        all_prefilled_blocks = kvcache_manager.get_prefilled_cpu_blocks(seq)
        total_prefill_tokens = kvcache_manager.get_prefill_len(seq)
        last_block_valid_tokens = total_prefill_tokens % block_size
        if last_block_valid_tokens == 0 and total_prefill_tokens > 0:
            last_block_valid_tokens = block_size

        last_prefilled_block = all_prefilled_blocks[-1] if all_prefilled_blocks else None
        selected_contains_last = (
            cpu_block_table and cpu_block_table[-1] == last_prefilled_block
        )
        effective_last_block_tokens = (
            last_block_valid_tokens if selected_contains_last else block_size
        )

        load_slots = offload_engine.decode_load_slots
        o_acc, lse_acc = self._decode_ring_buffer_pipeline(
            q_batched=q_batched,
            cpu_block_table=cpu_block_table,
            load_slots=load_slots,
            offload_engine=offload_engine,
            block_size=block_size,
            last_block_valid_tokens=effective_last_block_tokens,
            layer_id=layer_id,
            softmax_scale=softmax_scale,
        )

        seq_len = len(seq)
        decode_pos_in_block = (seq_len - 1) % block_size
        decode_start_pos = kvcache_manager.get_decode_start_pos(seq)
        decode_start_pos_in_block = decode_start_pos % block_size
        num_accumulated = decode_pos_in_block - decode_start_pos_in_block + 1

        compute_stream = offload_engine.compute_stream
        compute_stream.wait_stream(torch.cuda.default_stream())

        with torch.cuda.stream(compute_stream):
            if num_accumulated > 0:
                if getattr(offload_engine, "is_head_first", False):
                    decode_k = offload_engine.decode_k_buffer[
                        layer_id, :, decode_start_pos_in_block : decode_pos_in_block + 1
                    ].transpose(0, 1)
                    decode_v = offload_engine.decode_v_buffer[
                        layer_id, :, decode_start_pos_in_block : decode_pos_in_block + 1
                    ].transpose(0, 1)
                else:
                    decode_k = offload_engine.decode_k_buffer[
                        layer_id, decode_start_pos_in_block : decode_pos_in_block + 1
                    ]
                    decode_v = offload_engine.decode_v_buffer[
                        layer_id, decode_start_pos_in_block : decode_pos_in_block + 1
                    ]
                decode_k = decode_k.unsqueeze(0)
                decode_v = decode_v.unsqueeze(0)

                decode_o, decode_lse = _postrope_flash_attn_with_lse(
                    q_batched,
                    decode_k,
                    decode_v,
                    softmax_scale=softmax_scale,
                    causal=False,
                )

                if o_acc is None:
                    o_acc = decode_o
                else:
                    o_acc, _ = _postrope_merge_attention_outputs(
                        o_acc, lse_acc, decode_o, decode_lse
                    )

        if o_acc is None:
            raise RuntimeError("Chunked decode attention failed: no KV available")

        torch.cuda.default_stream().wait_stream(compute_stream)
        return o_acc

    def _decode_ring_buffer_pipeline(
        self,
        q_batched: torch.Tensor,
        cpu_block_table: List[int],
        load_slots: List[int],
        offload_engine: "OffloadEngine",
        block_size: int,
        last_block_valid_tokens: int,
        layer_id: int,
        softmax_scale: float,
    ):
        num_blocks = len(cpu_block_table)
        if num_blocks == 0 or not load_slots:
            return None, None

        o_acc, lse_acc = None, None
        num_slots = len(load_slots)
        compute_stream = offload_engine.compute_stream

        num_preload = min(num_slots, num_blocks)
        for i in range(num_preload):
            cpu_block_id = cpu_block_table[i]
            offload_engine.load_to_slot_layer(
                load_slots[i],
                layer_id,
                cpu_block_id,
                chunk_idx=cpu_block_id,
                is_prefill=False,
            )

        for block_idx in range(num_blocks):
            current_slot = load_slots[block_idx % num_slots]
            cpu_block_id = cpu_block_table[block_idx]
            offload_engine.wait_slot_layer(current_slot)

            with torch.cuda.stream(compute_stream):
                prev_k, prev_v = offload_engine.get_kv_for_slot(current_slot)
                is_last_block = block_idx == num_blocks - 1
                if is_last_block and last_block_valid_tokens < block_size:
                    prev_k = prev_k[:, :last_block_valid_tokens, :, :]
                    prev_v = prev_v[:, :last_block_valid_tokens, :, :]

                prev_o, prev_lse = _postrope_flash_attn_with_lse(
                    q_batched,
                    prev_k,
                    prev_v,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                offload_engine.record_slot_compute_done(current_slot)

            next_block_idx = block_idx + num_slots
            if next_block_idx < num_blocks:
                next_cpu_block_id = cpu_block_table[next_block_idx]
                offload_engine.load_to_slot_layer(
                    current_slot,
                    layer_id,
                    next_cpu_block_id,
                    chunk_idx=next_cpu_block_id,
                    is_prefill=False,
                )

            with torch.cuda.stream(compute_stream):
                if o_acc is None:
                    o_acc, lse_acc = prev_o, prev_lse
                else:
                    o_acc, lse_acc = _postrope_merge_attention_outputs(
                        o_acc, lse_acc, prev_o, prev_lse
                    )

        return o_acc, lse_acc

    # ---------------------------------------------------------------------
    # Offload hook
    # ---------------------------------------------------------------------

    def on_prefill_offload(
        self,
        cpu_block_id: int,
        layer_id: int,
        k_cache: torch.Tensor,
        num_valid_tokens: int,
    ) -> None:
        """Cache pooled post-RoPE K summaries on GPU for stage-1 selection."""
        if num_valid_tokens <= 0:
            return

        if k_cache.shape[0] == self._num_kv_heads:
            self._kvcache_block_size = int(k_cache.shape[1])
            k_block = k_cache[:, :num_valid_tokens, :].transpose(0, 1).contiguous()
        else:
            self._kvcache_block_size = int(k_cache.shape[0])
            k_block = k_cache[:num_valid_tokens].contiguous()

        pooled_k = self._mean_pool(k_block, self.ESTIMATE_CHUNK_SIZE).to(torch.float16)
        pooled_k_chunk = pooled_k.float().mean(dim=0)
        self_cos_k = self._compute_self_cosine(k_block, self.ESTIMATE_CHUNK_SIZE).to(torch.float32)
        self_cos_k_chunk = self_cos_k.mean(dim=0)
        self._k_summary_cache.setdefault(layer_id, {})[cpu_block_id] = (
            pooled_k_chunk.detach(),
            self_cos_k_chunk.detach(),
        )

    def __repr__(self) -> str:
        return "PostRoPEPolicy()"
