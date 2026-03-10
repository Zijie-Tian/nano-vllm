import torch
from typing import Tuple, Optional


def quantize_kcache_per_token(
    k_cache: torch.Tensor, bits: int = 2, sym: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-token quantization for K Cache (quantize along the head_dim dimension).
    """
    assert bits > 0 and bits <= 8, "bits must be between 1 and 8"

    if sym:
        q_max = (1 << (bits - 1)) - 1
        max_val = k_cache.abs().max(dim=-1, keepdim=True).values
        scales = max_val.clamp(min=1e-5) / q_max
        k_q = torch.round(k_cache / scales).clamp(-q_max, q_max).to(torch.int8)
        return k_q, scales, None
    else:
        q_min = -(1 << (bits - 1))
        q_max = (1 << (bits - 1)) - 1
        k_min = k_cache.min(dim=-1, keepdim=True).values
        k_max = k_cache.max(dim=-1, keepdim=True).values
        scales = (k_max - k_min) / (q_max - q_min)
        scales = scales.clamp(min=1e-8)
        zero_points = q_min * scales - k_min
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
    chunk_size_m: int = 32768,
) -> torch.Tensor:
    """
    High-performance bit-serial interleaving packing for T-MAC.
    Optimized with bitwise operators and memory-efficient chunking along the M dimension.

    Args:
        kv_q: [batch, n_head, M, K] (int8)
        chunk_size_m: Number of tokens to process per chunk to avoid OOM.
    """
    bias = 1 << (bits - 1)
    
    # 1. Handle V cache transpose if needed
    if not is_key:
        kv_q = kv_q.transpose(-2, -1).contiguous()

    batch, n_head, M, K = kv_q.shape
    ngroups_per_elem = 8 // g
    M_exp = M * bits
    
    # 2. Pre-allocate final output
    # Shape: [batch, n_head, M_exp//bm, K//g, bm//ngroups_per_elem]
    out = torch.empty((batch, n_head, M_exp // bm, K // g, bm // ngroups_per_elem), 
                     device=kv_q.device, dtype=torch.uint8)

    # 3. Process in chunks along the M dimension
    # Each chunk in 'out' corresponds to bm tokens in expanded space (M_exp).
    # So each chunk in 'out' handles bm/bits tokens in original space.
    m_per_out_tile = bm // bits
    
    # We must process in multiples of m_per_out_tile
    effective_chunk_m = (chunk_size_m // m_per_out_tile) * m_per_out_tile
    effective_chunk_m = max(effective_chunk_m, m_per_out_tile)

    for m_start in range(0, M, effective_chunk_m):
        m_end = min(m_start + effective_chunk_m, M)
        cur_m = m_end - m_start
        
        # Extract chunk and add bias
        kv_u_chunk = (kv_q[:, :, m_start:m_end, :] + bias).to(torch.uint8)
        
        # Step 1-3: Bitwise extraction and grouping
        w = kv_u_chunk.view(batch, n_head, cur_m, K // g, g)
        b0 = (w >> 0) & 1
        b1 = (w >> 1) & 1
        
        p0 = (b0[..., 0] << 0) | (b0[..., 1] << 1) | (b0[..., 2] << 2) | (b0[..., 3] << 3)
        p1 = (b1[..., 0] << 0) | (b1[..., 1] << 1) | (b1[..., 2] << 2) | (b1[..., 3] << 3)
        
        w_bit_serial = torch.stack([p0, p1], dim=3) # [B, H, cur_m, bits, K/g]
        
        # Step 4-5: T-MAC Interleaving Reshape
        simd_n_in, simd_n_out = 16, 8
        mgroup = ngroups_per_elem * simd_n_in
        
        w = w_bit_serial.view(batch, n_head, cur_m // simd_n_out, simd_n_out, bits, K // g).permute(0, 1, 2, 4, 3, 5)
        w = w.reshape(batch, n_head, (cur_m * bits) // mgroup, ngroups_per_elem, simd_n_in, K // g).permute(0, 1, 2, 4, 3, 5)
        
        cur_m_exp = cur_m * bits
        w = w.view(batch, n_head, cur_m_exp // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).permute(0, 1, 2, 6, 3, 7, 4, 5)
        
        # Step 5 Packing: Final 2 groups -> 1 byte
        res = (w[..., 0].to(torch.uint8) << 0) | (w[..., 1].to(torch.uint8) << 4)
        
        # 4. Fill output slice
        # Current chunk corresponds to tiles in range [m_start*bits//bm, m_end*bits//bm)
        out_start = (m_start * bits) // bm
        out_end = (m_end * bits) // bm
        out[:, :, out_start:out_end, :, :].copy_(res.reshape(batch, n_head, out_end - out_start, K // g, bm // ngroups_per_elem))
        
        # Cleanup chunk memory
        del kv_u_chunk, w, b0, b1, p0, p1, w_bit_serial, res
        
    return out
