"""
BLASST attention kernel with pre-computed skip mask.

This module implements the split-kernel stage:
1. `blasst_mask_forward` computes skip decisions.
2. `blasst_attn_forward` (this file) consumes that skip mask.

Design constraints implemented here:
- BLOCK_M = 64
- BLOCK_N = 64
- FP16 input tensors
- FP32 online-softmax accumulation for numerical stability
- GQA support (`num_heads` can be multiple of `num_kv_heads`)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# Fixed tile sizes from specification.
BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _blasst_attn_masked_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    skip_ptr,
    out_ptr,
    running_max_ptr,
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
    stride_sh,
    stride_sm,
    stride_sb,
    stride_oh,
    stride_om,
    stride_od,
    stride_rmh,
    stride_rmm,
    stride_lseh,
    stride_lsem,
    q_len,
    kv_len,
    head_dim,
    gqa_ratio,
    softmax_scale,
    granularity,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
):
    """
    One program computes one `(query_head, query_tile)` pair over all KV blocks.

    Tensor shapes:
        q:          [num_heads, q_len, head_dim]           fp16
        k:          [num_kv_heads, kv_len, head_dim]       fp16
        v:          [num_kv_heads, kv_len, head_dim]       fp16
        skip_mask:  [num_heads, q_len, num_kv_blocks]      bool/uint8
        running_max:[num_heads, q_len]                     fp32 (in/out)
        out:        [num_heads, q_len, head_dim]           fp16
        lse:        [num_heads, q_len]                     fp32
    """
    pid_m = tl.program_id(0)  # query tile index
    pid_h = tl.program_id(1)  # query head index

    # GQA mapping: multiple Q heads may share one KV head.
    kv_head_idx = pid_h // gqa_ratio

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < q_len
    d_mask = offs_d < head_dim

    # Load Q tile once and keep in registers. Compute is FP32 for stability.
    q_ptrs = (
        q_ptr
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    rm_ptrs = running_max_ptr + pid_h * stride_rmh + offs_m * stride_rmm
    m_i = tl.load(rm_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)

    # Online softmax state:
    # m_i: running max
    # l_i: running sum(exp(score - m_i))
    # acc: running sum(exp(score - m_i) * v)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kv_start = 0
    block_idx = 0
    while kv_start < kv_len:
        kv_end = tl.minimum(kv_start + granularity, kv_len)

        # Pass 1: compute local max over this granularity block.
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
            local_max = tl.maximum(local_max, tl.max(scores, axis=1))

            n_start += BLOCK_N

        local_max = tl.where(q_mask, local_max, -float("inf"))

        # Load pre-computed skip decision for this kv block.
        skip_ptrs = (
            skip_ptr
            + pid_h * stride_sh
            + offs_m * stride_sm
            + block_idx * stride_sb
        )
        skip = tl.load(skip_ptrs, mask=q_mask, other=1).to(tl.int32) != 0
        skip = skip | (~q_mask)

        needs_compute = (~skip) & q_mask
        any_compute = tl.sum(needs_compute.to(tl.int32), axis=0) > 0

        # Running max is updated for all rows (including skipped rows).
        new_m = tl.where(q_mask, tl.maximum(m_i, local_max), m_i)

        if any_compute:
            # Online-softmax rescale only for rows that actually compute PV.
            alpha = tl.where(needs_compute, tl.exp(m_i - new_m), 1.0)
            l_i = l_i * alpha
            acc = acc * alpha[:, None]

            # Pass 2: recompute scores and perform masked PV accumulation.
            safe_new_m = tl.where(needs_compute, new_m, 0.0)
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
                p = tl.where(needs_compute[:, None] & n_mask[None, :], p, 0.0)

                l_i += tl.sum(p, axis=1)
                acc += tl.dot(p, v)

                n_start += BLOCK_N

        m_i = new_m
        kv_start += granularity
        block_idx += 1

    # Final normalization. Rows with no computed mass (all skipped) return 0.
    denom = l_i[:, None]
    out = tl.where((denom > 0) & q_mask[:, None], acc / denom, 0.0)

    out_ptrs = (
        out_ptr
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(out_ptr.type.element_ty), mask=q_mask[:, None] & d_mask[None, :])

    tl.store(rm_ptrs, m_i, mask=q_mask)

    # LSE is used by downstream merge logic.
    # If all blocks were skipped for a row, l_i == 0 and we emit -inf.
    lse_i = tl.where((l_i > 0) & q_mask, m_i + tl.log(l_i), -float("inf"))
    lse_ptrs = lse_ptr + pid_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse_i, mask=q_mask)


def _select_block_d(head_dim: int) -> int:
    """Pick compile-time BLOCK_D with masked tail support."""
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    if head_dim <= 256:
        return 256
    raise ValueError(f"Unsupported head_dim={head_dim}. Expected <= 256.")


def blasst_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    skip_mask: torch.Tensor,
    running_max: torch.Tensor,
    softmax_scale: float,
    granularity: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BLASST attention forward pass with pre-computed skip mask.

    Args:
        q: [num_heads, q_len, head_dim], fp16
        k: [num_kv_heads, kv_len, head_dim], fp16
        v: [num_kv_heads, kv_len, head_dim], fp16
        skip_mask: [num_heads, q_len, num_kv_blocks], bool/uint8
        running_max: [num_heads, q_len], fp32 (input state)
        softmax_scale: scalar scale factor, usually 1 / sqrt(head_dim)
        granularity: kv block size used to define `num_kv_blocks`

    Returns:
        output: [num_heads, q_len, head_dim], fp16
        new_running_max: [num_heads, q_len], fp32
        lse: [num_heads, q_len], fp32
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be 3D tensors: [heads, seq, dim].")
    if skip_mask.ndim != 3:
        raise ValueError("skip_mask must be 3D tensor: [heads, q_len, num_kv_blocks].")
    if running_max.ndim != 2:
        raise ValueError("running_max must be 2D tensor: [heads, q_len].")

    if q.device.type != "cuda":
        raise ValueError("blasst_attn_forward requires CUDA tensors.")
    if k.device != q.device or v.device != q.device or skip_mask.device != q.device:
        raise ValueError("q, k, v, skip_mask must be on the same CUDA device.")
    if running_max.device != q.device:
        raise ValueError("running_max must be on the same CUDA device as q.")

    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype not in (torch.float16, torch.bfloat16) or v.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/k/v must be float16 or bfloat16. Got q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    if running_max.dtype != torch.float32:
        raise TypeError("running_max must be float32.")
    if skip_mask.dtype not in (torch.bool, torch.uint8):
        raise TypeError(f"skip_mask must be bool or uint8, got {skip_mask.dtype}.")
    if granularity <= 0:
        raise ValueError("granularity must be positive.")

    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, kv_dim = k.shape

    if v.shape != (num_kv_heads, kv_len, kv_dim):
        raise ValueError("v shape must match k shape.")
    if kv_dim != head_dim:
        raise ValueError("q/k/v head_dim must match.")
    if running_max.shape != (num_heads, q_len):
        raise ValueError("running_max shape must be [num_heads, q_len].")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA.")

    num_kv_blocks = triton.cdiv(kv_len, granularity)
    expected_mask_shape = (num_heads, q_len, num_kv_blocks)
    if skip_mask.shape != expected_mask_shape:
        raise ValueError(
            f"skip_mask shape mismatch. Expected {expected_mask_shape}, got {tuple(skip_mask.shape)}."
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    skip_u8 = skip_mask.to(torch.uint8).contiguous()

    out = torch.empty_like(q_c)
    new_running_max = running_max.contiguous().clone()
    lse = torch.empty((num_heads, q_len), device=q.device, dtype=torch.float32)

    block_d = _select_block_d(head_dim)
    gqa_ratio = num_heads // num_kv_heads
    num_warps = 4 if block_d <= 64 else 8

    grid = (triton.cdiv(q_len, BLOCK_M), num_heads)
    _blasst_attn_masked_kernel[grid](
        q_c,
        k_c,
        v_c,
        skip_u8,
        out,
        new_running_max,
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
        skip_u8.stride(0),
        skip_u8.stride(1),
        skip_u8.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        new_running_max.stride(0),
        new_running_max.stride(1),
        lse.stride(0),
        lse.stride(1),
        q_len,
        kv_len,
        head_dim,
        gqa_ratio,
        float(softmax_scale),
        int(granularity),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )

    return out, new_running_max, lse


def example_usage() -> None:
    """Minimal runnable example."""
    if not torch.cuda.is_available():
        print("CUDA is required for this example.")
        return

    torch.manual_seed(0)

    num_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    q_len = 256
    kv_len = 512
    head_dim = 128
    granularity = 128

    q = torch.randn(num_heads, q_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

    num_kv_blocks = triton.cdiv(kv_len, granularity)
    # Random skip mask example (False => compute PV, True => skip).
    skip_mask = (torch.rand(num_heads, q_len, num_kv_blocks, device="cuda") < 0.35)

    running_max = torch.full((num_heads, q_len), -float("inf"), device="cuda", dtype=torch.float32)

    out, new_running_max, lse = blasst_attn_forward(
        q=q,
        k=k,
        v=v,
        skip_mask=skip_mask,
        running_max=running_max,
        softmax_scale=1.0 / math.sqrt(head_dim),
        granularity=granularity,
    )

    print("output shape:", tuple(out.shape))
    print("new_running_max shape:", tuple(new_running_max.shape))
    print("lse shape:", tuple(lse.shape))
    print("all-skipped rows:", int((~torch.isfinite(lse)).sum().item()))


if __name__ == "__main__":
    example_usage()
