"""
Quantization Utilities - Weight Quantization and KV Cache Quantization

================================================================================
                           Quantization Concepts
================================================================================

1. Basic Quantization Principle:
   +-------------------------------------------------------------------------+
   |  Original (float)    Quantize     Quantized (int)                       |
   |  ----------------  ------------>  ---------------                       |
   |       x             q = round(x / scale)                                |
   |                                                                         |
   |  Quantized (int)   Dequantize    Reconstructed (float)                  |
   |  ----------------  ------------>  --------------------                  |
   |       q             x' = q * scale                                      |
   +-------------------------------------------------------------------------+

2. Symmetric vs Asymmetric Quantization:
   +-------------------------------------------------------------------------+
   |  Symmetric (sym=True):                                                  |
   |    scale = max(|x|) / q_max                                             |
   |    q = round(x / scale)                                                 |
   |    x' = q * scale                                                       |
   |                                                                         |
   |  Asymmetric (sym=False):                                                |
   |    scale = (max - min) / (q_max - q_min)                                |
   |    zero_point = q_min * scale - min                                     |
   |    q = round((x + zero_point) / scale)                                  |
   |    x' = q * scale - zero_point                                          |
   +-------------------------------------------------------------------------+

3. Weight Quantization Strategies:
   +-------------------------------------------------------------------------+
   |  per-tensor: single scale for entire matrix                             |
   |  +---------------------+                                                |
   |  | * * * * * * * * * * | --> scale                                      |
   |  | * * * * * * * * * * |                                                |
   |  | * * * * * * * * * * |                                                |
   |  +---------------------+                                                |
   |                                                                         |
   |  per-group: one scale per group along K dimension                       |
   |  +---------------------+                                                |
   |  | * * * * | * * * * | | --> scale0, scale1, ...                        |
   |  | * * * * | * * * * | |                                                |
   |  | * * * * | * * * * | |                                                |
   |  +---------------------+                                                |
   |    group0     group1                                                    |
   +-------------------------------------------------------------------------+

================================================================================
                      KV Cache Quantization (Chunk-based)
================================================================================

Terminology:
  - seq_len / chunk_size: number of tokens in a sequence/chunk
  - token: a single position in the sequence (one row in K/V matrix)
  - head_dim: dimension of each attention head (number of features per token)
  - channel: one feature dimension across all tokens (one column in K/V matrix)

4. K Cache Quantization - Per-Token (quantize along head_dim):
   +-------------------------------------------------------------------------+
   |  K_chunk: [chunk_size, head_dim]                                        |
   |           = [num_tokens, features_per_token]                            |
   |                                                                         |
   |  Each token (row) is quantized independently:                           |
   |  +-------------------------------------------+                          |
   |  | token0: [* * * * * * * * * * * *] -> s0   |  quantize direction -->  |
   |  | token1: [* * * * * * * * * * * *] -> s1   |                          |
   |  | token2: [* * * * * * * * * * * *] -> s2   |                          |
   |  | ...                                       |                          |
   |  | tokenN: [* * * * * * * * * * * *] -> sN   |                          |
   |  +-------------------------------------------+                          |
   |                                                                         |
   |  Output shapes:                                                         |
   |    K_q:      [chunk_size, head_dim]  (int8 quantized values)            |
   |    K_scales: [chunk_size, 1]         (one scale per token/row)          |
   +-------------------------------------------------------------------------+

5. V Cache Quantization - Per-Channel (quantize along seq_len):
   +-------------------------------------------------------------------------+
   |  V_chunk: [chunk_size, head_dim]                                        |
   |           = [num_tokens, features_per_token]                            |
   |                                                                         |
   |  Each channel (column) is quantized independently:                      |
   |  +-------------------------------------------+                          |
   |  |          c0   c1   c2   c3   c4           |  quantize                |
   |  |          s0   s1   s2   s3   s4  scales   |  direction               |
   |  | token0  [ *    *    *    *    *  ]        |      |                   |
   |  | token1  [ *    *    *    *    *  ]        |      |                   |
   |  | token2  [ *    *    *    *    *  ]        |      v                   |
   |  | ...     [ *    *    *    *    *  ]        |                          |
   |  | tokenN  [ *    *    *    *    *  ]        |                          |
   |  +-------------------------------------------+                          |
   |                                                                         |
   |  Output shapes:                                                         |
   |    V_q:      [chunk_size, head_dim]  (int8 quantized values)            |
   |    V_scales: [1, head_dim]           (one scale per channel/column)     |
   +-------------------------------------------------------------------------+

================================================================================
                              Function List
================================================================================

Basic Weight Quantization (compatible with test_e2e.py):
  - quantize_weight_per_tensor()    : Per-tensor quantization
  - quantize_weight_per_group()     : Per-group quantization
  - dequantize_weight_per_tensor()  : Per-tensor dequantization
  - dequantize_weight_per_group()   : Per-group dequantization

KV Cache Quantization - Single Chunk:
  - quantize_kcache_chunk()         : K cache single chunk (per-token)
  - quantize_vcache_chunk()         : V cache single chunk (per-channel)
  - dequantize_kcache_chunk()       : K cache single chunk dequantization
  - dequantize_vcache_chunk()       : V cache single chunk dequantization

KV Cache Quantization - Full (Multi-Chunk):
  - quantize_kcache_chunked()       : Full K cache with chunking (per-token)
  - quantize_vcache_chunked()       : Full V cache with chunking (per-channel)
  - dequantize_kcache_chunked()     : Full K cache dequantization
  - dequantize_vcache_chunked()     : Full V cache dequantization

Utility Functions:
  - compute_sqnr()                  : Compute Signal-to-Quantization-Noise Ratio
  - nmse()                          : Compute Normalized Mean Squared Error
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
#                        Basic Weight Quantization
# =============================================================================

def quantize_weight_per_tensor(
    weight: np.ndarray,
    bits: int,
    sym: bool = True
) -> Tuple[np.ndarray, float, Optional[float]]:
    """
    Per-tensor quantization: entire tensor uses a single scale.

    Use case: Uniform weight distribution, simple and efficient.

    Args:
        weight: Input weight matrix [M, K]
        bits: Quantization bit width (2, 3, 4, etc.)
        sym: True=symmetric, False=asymmetric quantization

    Returns:
        weight_q: Quantized weights (int8)
        scale: Quantization scale (float)
        zero_point: Zero point (None for symmetric, float for asymmetric)

    Example:
        >>> weight = np.random.randn(128, 256).astype(np.float16)
        >>> w_q, scale, zp = quantize_weight_per_tensor(weight, bits=4, sym=True)
        >>> print(f"scale={scale:.4f}, w_q.shape={w_q.shape}")
    """
    if sym:
        # Symmetric: q = round(x / scale), x' = q * scale
        q_max = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(weight))
        scale = max(scale, 1e-5) / q_max
        weight_q = np.round(weight / scale).clip(-q_max, q_max).astype(np.int8)
        return weight_q, scale, None
    else:
        # Asymmetric: q = round((x + zp) / scale), x' = q * scale - zp
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1

        w_min, w_max = np.min(weight), np.max(weight)
        scale = (w_max - w_min) / (q_max - q_min)
        scale = max(scale, 1e-8)
        zero_point = q_min * scale - w_min

        weight_q = np.round((weight + zero_point) / scale).clip(q_min, q_max).astype(np.int8)
        return weight_q, scale, zero_point


def quantize_weight_per_group(
    weight: np.ndarray,
    bits: int,
    group_size: int,
    sym: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Per-group quantization: groups along K dimension, each group has its own scale.

    Use case: Non-uniform weight distribution, higher precision needed.

    Data layout:
        weight: [M, K] -> grouped as [M, K//group_size, group_size]
        scales: [M, K//group_size]

    Args:
        weight: Input weight matrix [M, K]
        bits: Quantization bit width
        group_size: Number of elements per group (must divide K)
        sym: True=symmetric, False=asymmetric quantization

    Returns:
        weight_q: Quantized weights [M, K] (int8)
        scales: Per-group scales [M, K//group_size]
        zero_points: Per-group zero points (None for symmetric)

    Example:
        >>> weight = np.random.randn(128, 256).astype(np.float16)
        >>> w_q, scales, zp = quantize_weight_per_group(weight, bits=4, group_size=64)
        >>> print(f"scales.shape={scales.shape}")  # (128, 4)
    """
    M, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    n_groups = K // group_size
    weight_grouped = weight.reshape(M, n_groups, group_size)

    if sym:
        q_max = 2 ** (bits - 1) - 1

        # Compute per-group scales
        scales = np.max(np.abs(weight_grouped), axis=2)
        scales = np.maximum(scales, 1e-5) / q_max

        # Quantize each group
        weight_q = np.zeros_like(weight, dtype=np.int8)
        for i in range(n_groups):
            start, end = i * group_size, (i + 1) * group_size
            scale = scales[:, i:i+1]
            weight_q[:, start:end] = np.round(weight[:, start:end] / scale).clip(-q_max, q_max).astype(np.int8)

        return weight_q, scales, None
    else:
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1

        # Compute per-group min/max
        w_min = np.min(weight_grouped, axis=2)
        w_max = np.max(weight_grouped, axis=2)

        # Compute scales and zero_points
        scales = (w_max - w_min) / (q_max - q_min)
        scales = np.maximum(scales, 1e-8)
        zero_points = q_min * scales - w_min

        # Quantize each group
        weight_q = np.zeros_like(weight, dtype=np.int8)
        for i in range(n_groups):
            start, end = i * group_size, (i + 1) * group_size
            scale = scales[:, i:i+1]
            zp = zero_points[:, i:i+1]
            weight_q[:, start:end] = np.round((weight[:, start:end] + zp) / scale).clip(q_min, q_max).astype(np.int8)

        return weight_q, scales, zero_points


def dequantize_weight_per_tensor(
    weight_q: np.ndarray,
    scale: float,
    zero_point: Optional[float] = None,
    dtype: type = np.float16
) -> np.ndarray:
    """
    Per-tensor dequantization.

    Args:
        weight_q: Quantized weights (int8)
        scale: Quantization scale
        zero_point: Zero point (None for symmetric)
        dtype: Output data type

    Returns:
        weight: Dequantized weights
    """
    if zero_point is None:
        return weight_q.astype(dtype) * scale
    else:
        return weight_q.astype(dtype) * scale - zero_point


def dequantize_weight_per_group(
    weight_q: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    zero_points: Optional[np.ndarray] = None,
    dtype: type = np.float16
) -> np.ndarray:
    """
    Per-group dequantization.

    Args:
        weight_q: Quantized weights [M, K] (int8)
        scales: Per-group scales [M, K//group_size]
        group_size: Number of elements per group
        zero_points: Per-group zero points
        dtype: Output data type

    Returns:
        weight: Dequantized weights [M, K]
    """
    M, K = weight_q.shape
    n_groups = K // group_size

    weight = np.zeros((M, K), dtype=dtype)

    for i in range(n_groups):
        start, end = i * group_size, (i + 1) * group_size
        scale = scales[:, i:i+1]

        if zero_points is None:
            weight[:, start:end] = weight_q[:, start:end].astype(dtype) * scale
        else:
            zp = zero_points[:, i:i+1]
            weight[:, start:end] = weight_q[:, start:end].astype(dtype) * scale - zp

    return weight


# =============================================================================
#                     KV Cache Quantization (Chunk-based)
# =============================================================================

def quantize_kcache_chunk(
    k_chunk: np.ndarray,
    bits: int,
    sym: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    K Cache single chunk quantization - Per-Token (quantize along head_dim).

    Each row (token) in the chunk is quantized independently using its own scale.
    This is equivalent to per-group quantization with group_size = head_dim.

    Diagram:
        K_chunk: [chunk_size, head_dim] = [num_tokens, features_per_token]
        +-------------------------------------------+
        | token0: [* * * * * * * * * * * *] -> s0   |  quantize along row -->
        | token1: [* * * * * * * * * * * *] -> s1   |
        | token2: [* * * * * * * * * * * *] -> s2   |
        | ...                                       |
        +-------------------------------------------+

    Args:
        k_chunk: K cache chunk [chunk_size, head_dim]
                 chunk_size = number of tokens
                 head_dim = features per token
        bits: Quantization bit width
        sym: True=symmetric quantization

    Returns:
        k_q: Quantized K [chunk_size, head_dim] (int8)
        k_scales: Per-token scales [chunk_size, 1]
        k_zeros: Per-token zero points (None for symmetric)
    """
    chunk_size, head_dim = k_chunk.shape

    # Use per-group quantization with group_size=head_dim (one scale per row)
    k_q, k_scales, k_zeros = quantize_weight_per_group(
        k_chunk, bits=bits, group_size=head_dim, sym=sym
    )

    # k_scales shape: [chunk_size, 1]
    return k_q, k_scales, k_zeros


def quantize_vcache_chunk(
    v_chunk: np.ndarray,
    bits: int,
    sym: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    V Cache single chunk quantization - Per-Channel (quantize along seq_len).

    Each column (channel/feature) in the chunk is quantized independently.
    We transpose, apply per-group quantization, then transpose back.

    Diagram:
        V_chunk: [chunk_size, head_dim] = [num_tokens, features_per_token]
        +-------------------------------------------+
        |          c0   c1   c2   c3   c4           |
        |          s0   s1   s2   s3   s4  scales   |
        | token0  [ *    *    *    *    *  ]        |  quantize
        | token1  [ *    *    *    *    *  ]        |  along
        | token2  [ *    *    *    *    *  ]        |  column
        | ...     [ *    *    *    *    *  ]        |    |
        +-------------------------------------------+    v

    Args:
        v_chunk: V cache chunk [chunk_size, head_dim]
                 chunk_size = number of tokens
                 head_dim = features per token (channels)
        bits: Quantization bit width
        sym: True=symmetric quantization

    Returns:
        v_q: Quantized V [chunk_size, head_dim] (int8)
        v_scales: Per-channel scales [1, head_dim]
        v_zeros: Per-channel zero points (None for symmetric)
    """
    chunk_size, head_dim = v_chunk.shape

    # Transpose: [chunk_size, head_dim] -> [head_dim, chunk_size]
    v_t = v_chunk.T

    # Apply per-group quantization with group_size=chunk_size (one scale per column)
    v_q_t, v_scales_t, v_zeros_t = quantize_weight_per_group(
        v_t, bits=bits, group_size=chunk_size, sym=sym
    )

    # Transpose back:
    # v_q_t: [head_dim, chunk_size] -> v_q: [chunk_size, head_dim]
    # v_scales_t: [head_dim, 1] -> v_scales: [1, head_dim]
    v_q = v_q_t.T
    v_scales = v_scales_t.T
    v_zeros = v_zeros_t.T if v_zeros_t is not None else None

    return v_q, v_scales, v_zeros


def dequantize_kcache_chunk(
    k_q: np.ndarray,
    k_scales: np.ndarray,
    k_zeros: Optional[np.ndarray] = None,
    dtype: type = np.float16
) -> np.ndarray:
    """
    K Cache single chunk dequantization - Per-Token.

    Args:
        k_q: Quantized K [chunk_size, head_dim] (int8)
        k_scales: Per-token scales [chunk_size, 1]
        k_zeros: Per-token zero points
        dtype: Output data type

    Returns:
        k_chunk: Dequantized K [chunk_size, head_dim]
    """
    # k_scales: [chunk_size, 1] broadcasts to [chunk_size, head_dim]
    if k_zeros is None:
        return k_q.astype(dtype) * k_scales
    else:
        return k_q.astype(dtype) * k_scales - k_zeros


def dequantize_vcache_chunk(
    v_q: np.ndarray,
    v_scales: np.ndarray,
    v_zeros: Optional[np.ndarray] = None,
    dtype: type = np.float16
) -> np.ndarray:
    """
    V Cache single chunk dequantization - Per-Channel.

    Args:
        v_q: Quantized V [chunk_size, head_dim] (int8)
        v_scales: Per-channel scales [1, head_dim]
        v_zeros: Per-channel zero points
        dtype: Output data type

    Returns:
        v_chunk: Dequantized V [chunk_size, head_dim]
    """
    # v_scales: [1, head_dim] broadcasts to [chunk_size, head_dim]
    if v_zeros is None:
        return v_q.astype(dtype) * v_scales
    else:
        return v_q.astype(dtype) * v_scales - v_zeros


# =============================================================================
#                 Full KV Cache Quantization (Multi-Chunk)
# =============================================================================

def quantize_kcache_chunked(
    K_fp: np.ndarray,
    chunk_size: int,
    bits: int = 4,
    sym: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize full K cache by splitting into chunks (per-token quantization).

    Splits K cache along sequence dimension, quantizes each chunk with
    per-token quantization, then concatenates results.

    Diagram:
        K_fp: [batch, n_head, seq_len, head_dim]
        +-------+-------+-------+-----+-------+
        |chunk0 |chunk1 |chunk2 | ... |chunkN |  seq_len = n_chunks * chunk_size
        +-------+-------+-------+-----+-------+
            |       |       |           |
            v       v       v           v
          [quantize_kcache_chunk] x N  (per-token)
            |       |       |           |
            v       v       v           v
        +-------+-------+-------+-----+-------+
        | K_q_0 | K_q_1 | K_q_2 | ... | K_q_N |
        +-------+-------+-------+-----+-------+
        K_q: [batch, n_head, seq_len, head_dim]
        K_scales: [batch, n_head, seq_len, 1]  (one scale per token)

    Args:
        K_fp: Float K cache [batch, n_head, seq_len, head_dim]
        chunk_size: Size of each chunk (must divide seq_len)
        bits: Quantization bits
        sym: Symmetric quantization

    Returns:
        K_q: Quantized K cache [batch, n_head, seq_len, head_dim] (int8)
        K_scales: Per-token scales [batch, n_head, seq_len, 1]
    """
    batch, n_head, seq_len, head_dim = K_fp.shape
    assert seq_len % chunk_size == 0, \
        f"seq_len={seq_len} must be divisible by chunk_size={chunk_size}"

    n_chunks = seq_len // chunk_size
    K_q_all, K_scales_all = [], []

    for b in range(batch):
        K_q_batch, K_scales_batch = [], []
        for h in range(n_head):
            K_q_head, K_scales_head = [], []
            for c in range(n_chunks):
                start, end = c * chunk_size, (c + 1) * chunk_size
                k_chunk = K_fp[b, h, start:end, :]
                k_q, k_scales, _ = quantize_kcache_chunk(k_chunk, bits=bits, sym=sym)
                K_q_head.append(k_q)
                K_scales_head.append(k_scales)
            K_q_batch.append(np.concatenate(K_q_head, axis=0))
            K_scales_batch.append(np.concatenate(K_scales_head, axis=0))
        K_q_all.append(np.stack(K_q_batch, axis=0))
        K_scales_all.append(np.stack(K_scales_batch, axis=0))

    return np.stack(K_q_all, axis=0), np.stack(K_scales_all, axis=0)


def quantize_vcache_chunked(
    V_fp: np.ndarray,
    chunk_size: int,
    bits: int = 4,
    sym: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize full V cache by splitting into chunks (per-channel quantization).

    Splits V cache along sequence dimension, quantizes each chunk with
    per-channel quantization (each channel has its own scale within chunk).

    Diagram:
        V_fp: [batch, n_head, seq_len, head_dim]
        +-------+-------+-------+-----+-------+
        |chunk0 |chunk1 |chunk2 | ... |chunkN |
        +-------+-------+-------+-----+-------+
            |       |       |           |
            v       v       v           v
          [quantize_vcache_chunk] x N  (per-channel)
            |       |       |           |
            v       v       v           v
        scales:[1,hd] [1,hd] [1,hd]   [1,hd]  (one scale per channel per chunk)
        +-------+-------+-------+-----+-------+
        | V_q_0 | V_q_1 | V_q_2 | ... | V_q_N |
        +-------+-------+-------+-----+-------+
        V_q: [batch, n_head, seq_len, head_dim]
        V_scales: [batch, n_head, n_chunks, head_dim]

    Note: V_scales has n_chunks in dim2 (not seq_len) because per-channel
    quantization gives one scale per channel per chunk.

    Args:
        V_fp: Float V cache [batch, n_head, seq_len, head_dim]
        chunk_size: Size of each chunk (must divide seq_len)
        bits: Quantization bits
        sym: Symmetric quantization

    Returns:
        V_q: Quantized V cache [batch, n_head, seq_len, head_dim] (int8)
        V_scales: Per-channel per-chunk scales [batch, n_head, n_chunks, head_dim]
    """
    batch, n_head, seq_len, head_dim = V_fp.shape
    assert seq_len % chunk_size == 0, \
        f"seq_len={seq_len} must be divisible by chunk_size={chunk_size}"

    n_chunks = seq_len // chunk_size
    V_q_all, V_scales_all = [], []

    for b in range(batch):
        V_q_batch, V_scales_batch = [], []
        for h in range(n_head):
            V_q_head, V_scales_head = [], []
            for c in range(n_chunks):
                start, end = c * chunk_size, (c + 1) * chunk_size
                v_chunk = V_fp[b, h, start:end, :]
                v_q, v_scales, _ = quantize_vcache_chunk(v_chunk, bits=bits, sym=sym)
                V_q_head.append(v_q)
                V_scales_head.append(v_scales)  # [1, head_dim]
            V_q_batch.append(np.concatenate(V_q_head, axis=0))
            V_scales_batch.append(np.concatenate(V_scales_head, axis=0))  # [n_chunks, head_dim]
        V_q_all.append(np.stack(V_q_batch, axis=0))
        V_scales_all.append(np.stack(V_scales_batch, axis=0))

    return np.stack(V_q_all, axis=0), np.stack(V_scales_all, axis=0)


def dequantize_kcache_chunked(
    K_q: np.ndarray,
    K_scales: np.ndarray,
    chunk_size: int,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Dequantize full K cache by processing chunks.

    Calls dequantize_kcache_chunk for each chunk.

    Args:
        K_q: Quantized K cache [batch, n_head, seq_len, head_dim] (int8)
        K_scales: Per-token scales [batch, n_head, seq_len, 1]
        chunk_size: Size of each chunk
        dtype: Output data type

    Returns:
        K_dq: Dequantized K cache [batch, n_head, seq_len, head_dim]
    """
    batch, n_head, seq_len, head_dim = K_q.shape
    n_chunks = seq_len // chunk_size
    K_dq_all = []

    for b in range(batch):
        K_dq_batch = []
        for h in range(n_head):
            K_dq_head = []
            for c in range(n_chunks):
                start, end = c * chunk_size, (c + 1) * chunk_size
                k_chunk_q = K_q[b, h, start:end, :]
                k_chunk_scales = K_scales[b, h, start:end, :]
                k_chunk_dq = dequantize_kcache_chunk(k_chunk_q, k_chunk_scales, dtype=dtype)
                K_dq_head.append(k_chunk_dq)
            K_dq_batch.append(np.concatenate(K_dq_head, axis=0))
        K_dq_all.append(np.stack(K_dq_batch, axis=0))

    return np.stack(K_dq_all, axis=0)


def dequantize_vcache_chunked(
    V_q: np.ndarray,
    V_scales: np.ndarray,
    chunk_size: int,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Dequantize full V cache by processing chunks.

    Calls dequantize_vcache_chunk for each chunk.

    Args:
        V_q: Quantized V cache [batch, n_head, seq_len, head_dim] (int8)
        V_scales: Per-channel per-chunk scales [batch, n_head, n_chunks, head_dim]
        chunk_size: Size of each chunk
        dtype: Output data type

    Returns:
        V_dq: Dequantized V cache [batch, n_head, seq_len, head_dim]
    """
    batch, n_head, seq_len, head_dim = V_q.shape
    n_chunks = seq_len // chunk_size
    V_dq_all = []

    for b in range(batch):
        V_dq_batch = []
        for h in range(n_head):
            V_dq_head = []
            for c in range(n_chunks):
                start, end = c * chunk_size, (c + 1) * chunk_size
                v_chunk_q = V_q[b, h, start:end, :]
                v_chunk_scales = V_scales[b, h, c:c+1, :]  # [1, head_dim]
                v_chunk_dq = dequantize_vcache_chunk(v_chunk_q, v_chunk_scales, dtype=dtype)
                V_dq_head.append(v_chunk_dq)
            V_dq_batch.append(np.concatenate(V_dq_head, axis=0))
        V_dq_all.append(np.stack(V_dq_batch, axis=0))

    return np.stack(V_dq_all, axis=0)


# =============================================================================
#                            Utility Functions
# =============================================================================

def compute_sqnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio (SQNR).

    SQNR = 10 * log10(signal_power / noise_power)

    Args:
        original: Original signal
        reconstructed: Reconstructed signal after quantization

    Returns:
        SQNR value in dB (higher is better)

    Typical values:
        - SQNR > 30 dB: Excellent
        - SQNR 20-30 dB: Good
        - SQNR < 20 dB: Poor
    """
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise = original.astype(np.float64) - reconstructed.astype(np.float64)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)


def nmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error (NMSE).

    NMSE = ||original - reconstructed||^2 / ||original||^2

    Args:
        original: Original signal
        reconstructed: Reconstructed signal

    Returns:
        NMSE value (lower is better, 0 means identical)
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    noise = original - reconstructed
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)

    if signal_power < 1e-10:
        return float('inf')

    return noise_power / signal_power