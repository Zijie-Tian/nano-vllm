"""
Fused BLASST attention kernel.

This module implements BLASST's key optimization in a single Triton kernel:
1. Compute QK and local max for each query
2. Apply skip condition: local_max - running_max < ln_lambda
3. Only load V / compute PV when any query in the tile needs it

The kernel uses online softmax accumulation in FP32 for numerical stability.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# Fixed tile sizes requested by specification.
BLOCK_M: tl.constexpr = 64
BLOCK_N: tl.constexpr = 64


@triton.jit
def _blasst_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    running_max_ptr,
    block_skipped_ptr,
    lse_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    stride_rmh,
    stride_rmm,
    stride_bsh,
    stride_bsm,
    stride_lseh,
    stride_lsem,
    num_heads,
    num_kv_heads,
    q_len,
    kv_len,
    head_dim,
    gqa_ratio,
    softmax_scale,
    ln_lambda,
    granularity,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
):
    """
    One program handles one (head, query-tile) pair across all KV tokens.

    Grid:
        pid_m -> query tile index
        pid_h -> query head index

    Shapes:
        Q: [num_heads, q_len, head_dim]
        K: [num_kv_heads, kv_len, head_dim]
        V: [num_kv_heads, kv_len, head_dim]
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Map query head to KV head (GQA).
    kv_head_idx = pid_h // gqa_ratio

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < q_len
    d_mask = offs_d < head_dim

    # Load Q tile into registers (FP32 compute).
    q_ptrs = (
        q_ptr
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    # Load running max from input state.
    rm_ptrs = running_max_ptr + pid_h * stride_rmh + offs_m * stride_rmm
    m_i = tl.load(rm_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)

    # Online softmax running stats.
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Track if a query ever computed PV in this call.
    did_compute = tl.zeros([BLOCK_M], dtype=tl.int32)

    # Iterate KV at user granularity. Inside each granularity block,
    # we process K/V tiles of BLOCK_N=64 to respect shared-memory budget.
    kv_start = 0
    while kv_start < kv_len:
        kv_end = tl.minimum(kv_start + granularity, kv_len)

        # -------- Pass 1: QK-only to get local_max --------
        local_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        n_start = kv_start
        while n_start < kv_end:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < kv_end

            k_ptrs = (
                k_ptr
                + kv_head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            scores = tl.dot(q, tl.trans(k)) * softmax_scale
            scores = tl.where(n_mask[None, :], scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            local_max = tl.maximum(local_max, row_max)

            n_start += BLOCK_N

        local_max = tl.where(q_mask, local_max, -float("inf"))

        # BLASST skip condition (per query), but PV is computed for the whole
        # tile whenever any valid query needs this granularity block.
        skip_mask = (local_max - m_i) < ln_lambda
        skip_mask = skip_mask | (~q_mask)

        # If all valid rows skip, avoid loading V entirely.
        needs_pv = (~skip_mask) & q_mask
        any_pv = tl.sum(needs_pv.to(tl.int32), axis=0) > 0

        if any_pv:
            # Match reference behavior: once this granularity block is taken,
            # all valid rows participate in online softmax/PV accumulation.
            new_m = tl.where(q_mask, tl.maximum(m_i, local_max), m_i)

            # Rescale running accumulators where max changed.
            alpha = tl.where(q_mask, tl.exp(m_i - new_m), 1.0)
            l_i = l_i * alpha
            acc = acc * alpha[:, None]

            # -------- Pass 2: Recompute scores + PV --------
            safe_new_m = tl.where(q_mask, new_m, 0.0)
            n_start = kv_start
            while n_start < kv_end:
                offs_n = n_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < kv_end

                k_ptrs = (
                    k_ptr
                    + kv_head_idx * stride_kh
                    + offs_n[:, None] * stride_kn
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + kv_head_idx * stride_vh
                    + offs_n[:, None] * stride_vn
                    + offs_d[None, :] * stride_vd
                )

                k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

                scores = tl.dot(q, tl.trans(k)) * softmax_scale
                scores = tl.where(n_mask[None, :], scores, -float("inf"))

                p = tl.exp(scores - safe_new_m[:, None])
                p = tl.where(q_mask[:, None] & n_mask[None, :], p, 0.0)

                l_i += tl.sum(p, axis=1)
                acc += tl.dot(p, v)

                n_start += BLOCK_N

            m_i = new_m
            did_compute = did_compute | q_mask.to(tl.int32)
        else:
            # If entire tile skips this granularity block, update running max as requested.
            m_i = tl.where(q_mask, tl.maximum(m_i, local_max), m_i)

        kv_start += granularity

    # Final normalize; queries with zero mass (all skipped) return zeros.
    denom = l_i[:, None]
    denom_safe = tl.where(denom > 0, denom, 1.0)
    out = tl.where((denom > 0) & q_mask[:, None], acc / denom_safe, 0.0)

    out_ptrs = (
        out_ptr
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(out_ptr.type.element_ty), mask=q_mask[:, None] & d_mask[None, :])

    # Write updated running max and per-query skipped flag.
    tl.store(rm_ptrs, m_i, mask=q_mask)

    # block_skipped=True means this query never computed PV in this call.
    bs_ptrs = block_skipped_ptr + pid_h * stride_bsh + offs_m * stride_bsm
    tl.store(bs_ptrs, (did_compute == 0).to(tl.int8), mask=q_mask)

    # Per-call LSE for numerically stable cross-call merging.
    # Use a large negative finite sentinel (instead of -inf) for zero-mass rows
    # to keep downstream merge kernels stable when both sides have no mass.
    lse_i = tl.where((l_i > 0) & q_mask, m_i + tl.log(l_i), -1.0e30)
    lse_ptrs = lse_ptr + pid_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse_i, mask=q_mask)


def _select_block_d(head_dim: int) -> int:
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    if head_dim <= 256:
        return 256
    raise ValueError(f"Unsupported head_dim={head_dim}. Expected <= 256.")


def blasst_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    running_max: torch.Tensor,
    ln_lambda: float,
    softmax_scale: float,
    granularity: int = 128,
    return_lse: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Fused BLASST forward pass with conditional PV computation.

    Args:
        q: [num_heads, q_len, head_dim], fp16/bf16
        k: [num_kv_heads, kv_len, head_dim], fp16/bf16
        v: [num_kv_heads, kv_len, head_dim], fp16/bf16
        running_max: [num_heads, q_len], fp32 input/output running max
        ln_lambda: Skip threshold in log-space
        softmax_scale: Typically 1 / sqrt(head_dim)
        granularity: KV processing granularity (default 128)

    Returns:
        output: [num_heads, q_len, head_dim]
        new_running_max: [num_heads, q_len]
        block_skipped: [num_heads, q_len] bool; True means query never did PV
        lse (optional, when return_lse=True): [num_heads, q_len] in ln-space
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be 3D tensors: [heads, seq, dim].")
    if running_max.ndim != 2:
        raise ValueError("running_max must be 2D tensor: [heads, seq].")
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise ValueError("blasst_fused_forward requires CUDA tensors.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q dtype must be fp16/bf16, got {q.dtype}.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("k and v must have the same dtype as q.")
    if running_max.dtype != torch.float32:
        raise TypeError("running_max must be float32 for numerical stability.")

    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, kv_dim = k.shape

    if v.shape != (num_kv_heads, kv_len, kv_dim):
        raise ValueError("v shape must match k shape on [num_kv_heads, kv_len, head_dim].")
    if kv_dim != head_dim:
        raise ValueError("q/k/v head_dim must match.")
    if running_max.shape != (num_heads, q_len):
        raise ValueError("running_max shape must be [num_heads, q_len].")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA.")
    if granularity <= 0 or granularity % BLOCK_N != 0:
        raise ValueError(f"granularity must be a positive multiple of {BLOCK_N}.")

    # Keep memory accesses coalesced and predictable.
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    out = torch.empty_like(q_c)
    new_running_max = running_max.contiguous().clone()
    block_skipped_u8 = torch.empty((num_heads, q_len), device=q.device, dtype=torch.uint8)
    lse = torch.empty((num_heads, q_len), device=q.device, dtype=torch.float32)

    block_d = _select_block_d(head_dim)
    gqa_ratio = num_heads // num_kv_heads

    # Heuristic warp count for head_dim.
    num_warps = 4 if block_d <= 64 else 8

    grid = (triton.cdiv(q_len, BLOCK_M), num_heads)

    _blasst_fused_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        new_running_max,
        block_skipped_u8,
        lse,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        new_running_max.stride(0),
        new_running_max.stride(1),
        block_skipped_u8.stride(0),
        block_skipped_u8.stride(1),
        lse.stride(0),
        lse.stride(1),
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
        gqa_ratio,
        float(softmax_scale),
        float(ln_lambda),
        int(granularity),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )

    block_skipped = block_skipped_u8.bool()
    if return_lse:
        return out, new_running_max, block_skipped, lse
    return out, new_running_max, block_skipped


def example_usage() -> None:
    """Minimal runnable example."""
    if not torch.cuda.is_available():
        print("CUDA is required for this example.")
        return

    torch.manual_seed(0)

    num_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    q_len = 512
    kv_len = 512
    head_dim = 128

    q = torch.randn(num_heads, q_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

    running_max = torch.full((num_heads, q_len), -float("inf"), device="cuda", dtype=torch.float32)

    out, new_running_max, block_skipped = blasst_fused_forward(
        q=q,
        k=k,
        v=v,
        running_max=running_max,
        ln_lambda=math.log(0.5),
        softmax_scale=1.0 / math.sqrt(head_dim),
        granularity=128,
    )

    print("output shape:", tuple(out.shape))
    print("running_max shape:", tuple(new_running_max.shape))
    print("skip ratio:", block_skipped.float().mean().item())


if __name__ == "__main__":
    example_usage()
