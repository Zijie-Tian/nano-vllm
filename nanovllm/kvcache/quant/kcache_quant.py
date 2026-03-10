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
