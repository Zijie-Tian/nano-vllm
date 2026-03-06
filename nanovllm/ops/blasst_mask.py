"""
BLASST mask kernel.

This module computes the BLASST skip mask:
    skip = (local_max - running_max) < ln_lambda

For each (head, query, kv_block) pair, the kernel:
1. Computes local_max over the current KV granularity block.
2. Emits one skip flag.
3. Updates running_max online for the next block decision.

The implementation follows the requested constraints:
- BLOCK_M = 64 (query tile)
- BLOCK_N = 64 (KV tile used inside each granularity block)
- FP16 input tensors, FP32 accumulation
- GQA support (num_heads can be multiple of num_kv_heads)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# Fixed tile sizes from the design specification.
BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _blasst_mask_kernel(
    q_ptr,
    k_ptr,
    running_max_ptr,
    mask_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_rmh,
    stride_rmm,
    stride_mh,
    stride_mm,
    stride_mb,
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
    One Triton program handles one (query_head, query_tile) pair.

    Grid:
        pid_m -> query tile index
        pid_h -> query head index

    Tensor shapes:
        q:           [num_heads, q_len, head_dim]       (fp16)
        k:           [num_kv_heads, kv_len, head_dim]   (fp16)
        running_max: [num_heads, q_len]                 (fp32, read-only here)
        mask:        [num_heads, q_len, num_kv_blocks]  (uint8/bool-like)
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # GQA mapping: multiple Q heads may share one KV head.
    kv_head_idx = pid_h // gqa_ratio

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < q_len
    d_mask = offs_d < head_dim

    # Load one Q tile once; reused for all KV blocks in this program.
    q_ptrs = (
        q_ptr
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    # Running max is kept in registers and updated online across KV blocks.
    rm_ptrs = running_max_ptr + pid_h * stride_rmh + offs_m * stride_rmm
    m_i = tl.load(rm_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)

    kv_start = 0
    block_idx = 0
    while kv_start < kv_len:
        kv_end = tl.minimum(kv_start + granularity, kv_len)

        # Pass over this granularity block in BLOCK_N chunks to control SRAM usage.
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

            # Compute scores for this 64-token K tile.
            scores = tl.dot(q, tl.trans(k)) * softmax_scale
            scores = tl.where(n_mask[None, :], scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            local_max = tl.maximum(local_max, row_max)

            n_start += BLOCK_N

        # Skip decision is per query row in the tile.
        skip = (local_max - m_i) < ln_lambda
        skip = skip | (~q_mask)

        mask_ptrs = (
            mask_ptr
            + pid_h * stride_mh
            + offs_m * stride_mm
            + block_idx * stride_mb
        )
        tl.store(mask_ptrs, skip.to(mask_ptr.type.element_ty), mask=q_mask)

        # Online update is required so later block decisions use fresh running_max.
        m_i = tl.where(q_mask, tl.maximum(m_i, local_max), m_i)

        kv_start += granularity
        block_idx += 1


def _select_block_d(head_dim: int) -> int:
    """Pick compile-time BLOCK_D to cover head_dim with masked tail handling."""
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    if head_dim <= 256:
        return 256
    raise ValueError(f"Unsupported head_dim={head_dim}. Expected <= 256.")


def blasst_mask_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    running_max: torch.Tensor,
    ln_lambda: float,
    softmax_scale: float,
    granularity: int = 128,
) -> torch.Tensor:
    """
    Compute BLASST skip mask.

    Args:
        q: [num_heads, q_len, head_dim], fp16
        k: [num_kv_heads, kv_len, head_dim], fp16
        running_max: [num_heads, q_len], fp32 (read-only input state)
        ln_lambda: skip threshold in log-space
        softmax_scale: usually 1 / sqrt(head_dim)
        granularity: KV block size for mask decisions (default 128)

    Returns:
        skip_mask: [num_heads, q_len, num_kv_blocks], bool
            True  -> skip PV for this (head, query, kv_block)
            False -> keep PV computation for this block
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must be 3D tensors: [heads, seq, dim].")
    if running_max.ndim != 2:
        raise ValueError("running_max must be 2D tensor: [heads, seq].")
    if q.device.type != "cuda" or k.device.type != "cuda" or running_max.device.type != "cuda":
        raise ValueError("blasst_mask_forward requires CUDA tensors.")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/k must be float16 or bfloat16. Got q={q.dtype}, k={k.dtype}.")
    if running_max.dtype != torch.float32:
        raise TypeError("running_max must be float32 for numerical stability.")
    if granularity <= 0:
        raise ValueError("granularity must be positive.")

    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, kv_dim = k.shape
    if kv_dim != head_dim:
        raise ValueError("q and k must have the same head_dim.")
    if running_max.shape != (num_heads, q_len):
        raise ValueError("running_max shape must be [num_heads, q_len].")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA.")

    num_kv_blocks = triton.cdiv(kv_len, granularity)
    if num_kv_blocks == 0:
        return torch.empty((num_heads, q_len, 0), device=q.device, dtype=torch.bool)

    # Contiguous layout keeps stride math simple and memory accesses predictable.
    q_c = q.contiguous()
    k_c = k.contiguous()
    rm_c = running_max.contiguous()

    # Store as uint8 in-kernel, then expose bool to Python caller.
    skip_mask_u8 = torch.empty(
        (num_heads, q_len, num_kv_blocks), device=q.device, dtype=torch.uint8
    )

    block_d = _select_block_d(head_dim)
    gqa_ratio = num_heads // num_kv_heads
    num_warps = 4 if block_d <= 64 else 8

    grid = (triton.cdiv(q_len, BLOCK_M), num_heads)

    _blasst_mask_kernel[grid](
        q_c,
        k_c,
        rm_c,
        skip_mask_u8,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        rm_c.stride(0),
        rm_c.stride(1),
        skip_mask_u8.stride(0),
        skip_mask_u8.stride(1),
        skip_mask_u8.stride(2),
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

    return skip_mask_u8.bool()


def example_usage() -> None:
    """Minimal runnable example for quick manual validation."""
    if not torch.cuda.is_available():
        print("CUDA is required for this example.")
        return

    torch.manual_seed(0)

    num_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    q_len = 512
    kv_len = 1024
    head_dim = 128

    q = torch.randn(num_heads, q_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    running_max = torch.full((num_heads, q_len), -float("inf"), device="cuda", dtype=torch.float32)

    skip_mask = blasst_mask_forward(
        q=q,
        k=k,
        running_max=running_max,
        ln_lambda=math.log(0.5),
        softmax_scale=1.0 / math.sqrt(head_dim),
        granularity=128,
    )

    print("skip_mask shape:", tuple(skip_mask.shape))
    print("skip ratio:", skip_mask.float().mean().item())


if __name__ == "__main__":
    example_usage()
