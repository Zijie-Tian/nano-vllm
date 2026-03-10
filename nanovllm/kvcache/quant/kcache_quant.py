import torch
from typing import Tuple, Optional


def quantize_kcache_per_token(
    k_cache: torch.Tensor, bits: int = 2, sym: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-token quantization for K Cache (quantize along the head_dim dimension).

    Args:
        k_cache: Float K cache tensor, typically [..., seq_len, head_dim]
        bits: Quantization bits (default 2)
        sym: Symmetric quantization (True) or Asymmetric (False).
             For 2-bit, asymmetric allows utilizing the full 4 values [-2, 1]
             while symmetric uses 3 values [-1, 1].

    Returns:
        k_q: Quantized K cache tensor (int8), same shape as k_cache
        scales: Per-token scales, shape [..., seq_len, 1]
        zero_points: Per-token zero points (None if sym=True), shape [..., seq_len, 1]
    """
    assert bits > 0 and bits <= 8, "bits must be between 1 and 8"

    if sym:
        # Symmetric quantization
        q_max = (1 << (bits - 1)) - 1

        # Max along head_dim
        # scales: [..., seq_len, 1]
        max_val = k_cache.abs().max(dim=-1, keepdim=True).values
        scales = max_val.clamp(min=1e-5) / q_max

        # Quantize
        k_q = torch.round(k_cache / scales).clamp(-q_max, q_max).to(torch.int8)
        return k_q, scales, None
    else:
        # Asymmetric quantization
        q_min = -(1 << (bits - 1))
        q_max = (1 << (bits - 1)) - 1

        # Min, Max along head_dim
        k_min = k_cache.min(dim=-1, keepdim=True).values
        k_max = k_cache.max(dim=-1, keepdim=True).values

        # scales: [..., seq_len, 1]
        scales = (k_max - k_min) / (q_max - q_min)
        scales = scales.clamp(min=1e-8)

        # zero_points: [..., seq_len, 1]
        zero_points = q_min * scales - k_min

        # Quantize
        k_q = (
            torch.round((k_cache + zero_points) / scales)
            .clamp(q_min, q_max)
            .to(torch.int8)
        )
        return k_q, scales, zero_points


def dequantize_kcache_per_token(
    k_q: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize per-token quantized K Cache.

    Args:
        k_q: Quantized K cache (int8)
        scales: Per-token scales
        zero_points: Per-token zero points (None if symmetric)
        dtype: Output data type

    Returns:
        k_dq: Dequantized K cache tensor
    """
    k_q_f = k_q.to(dtype)
    if zero_points is None:
        return k_q_f * scales.to(dtype)
    else:
        return k_q_f * scales.to(dtype) - zero_points.to(dtype)


def pack_kvcache_tmac(
    kv_q: torch.Tensor,
    is_key: bool = True,
    bits: int = 2,
    g: int = 4,
    bm: int = 256,
    kfactor: int = 16,
    simd_n_in: int = 16,
    simd_n_out: int = 8,
) -> torch.Tensor:
    """
    Offline preprocess the KV cache before T-MAC inference using PyTorch.
    Follows the logic of preprocess_weights in model_utils.py.

    Args:
        kv_q: Quantized KV cache [batch, n_head, chunk_size, head_dim] (int8)
        is_key: True for K cache, False for V cache
        bits: Quantization bits (2, 4)
        g: LUT group size (default 4)
        bm: Tiling size for M (default 256)
        kfactor: Tiling size for K (default 16)
        simd_n_in: SIMD input lanes (16)
        simd_n_out: SIMD output lanes (8)

    Returns:
        packed_kv: Packed KV cache tensor (uint8)
    """
    # 1. Convert to uint8 (add bias 2^(bits-1))
    bias = 1 << (bits - 1)
    kv_u = (kv_q + bias).to(torch.uint8)

    if not is_key:
        # V cache: transpose chunk_size and head_dim
        # kv_u shape: [batch, n_head, head_dim, chunk_size]
        kv_u = kv_u.transpose(-2, -1).contiguous()

    batch, n_head, M, K = kv_u.shape
    ngroups_per_elem = 8 // g

    # Step 1 - Extract individual bits
    # [batch, n_head, M, K, bits]
    w = torch.stack([(kv_u >> ib) & 1 for ib in range(bits)], dim=-1)

    # Step 2 - Reorganize bits
    # [batch, n_head, M, bits, K//g, g]
    w = w.view(batch, n_head, M, K // g, g, bits).permute(0, 1, 2, 5, 3, 4)

    # Step 3 - Pack groups into LUT indices
    # [batch, n_head, M, bits, K//g]
    powers = torch.tensor(
        [1 << ig for ig in range(g)], device=w.device, dtype=torch.uint8
    )
    w = (w * powers.view(1, 1, 1, 1, 1, g)).sum(dim=-1, dtype=torch.uint8)

    # Step 4 - Reshape for SIMD processing
    # M_expanded = M * bits
    w = w.view(batch, n_head, M // simd_n_out, simd_n_out, bits, K // g).permute(
        0, 1, 2, 4, 3, 5
    )

    mgroup = ngroups_per_elem * simd_n_in
    w = w.reshape(
        batch, n_head, (M * bits) // mgroup, ngroups_per_elem, simd_n_in, K // g
    ).permute(0, 1, 2, 4, 3, 5)

    # Step 5 - Final tiling
    M_exp = M * bits
    w = w.view(
        batch,
        n_head,
        M_exp // bm,
        bm // mgroup,
        simd_n_in,
        ngroups_per_elem,
        K // g // kfactor,
        kfactor,
    ).permute(0, 1, 2, 6, 3, 7, 4, 5)

    ng_powers = torch.tensor(
        [1 << (ng * g) for ng in range(ngroups_per_elem)],
        device=w.device,
        dtype=torch.uint8,
    )
    w = (w * ng_powers.view(1, 1, 1, 1, 1, 1, 1, ngroups_per_elem)).sum(
        dim=-1, dtype=torch.uint8
    )

    # Final reshape to match TVM API: [batch, n_head, M_exp//bm, K//g, bm//ngroups_per_elem]
    w = w.reshape(batch, n_head, M_exp // bm, K // g, bm // ngroups_per_elem)

    return w

