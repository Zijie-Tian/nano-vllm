"""
Chunked attention implementation for CPU KV cache offloading.

This module implements flash attention with LSE (log-sum-exp) output,
enabling proper online softmax merging for chunked prefill.

Key functions:
- flash_attn_with_lse: Flash attention that returns output and LSE
- merge_attention_outputs: Merge outputs from multiple KV chunks
- chunked_prefill_attention: High-level interface for chunked attention
"""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple, List, Optional


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_with_lse(
    Q,
    K,
    V,
    Out,
    Lse,
    TMP,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention forward kernel with LSE output for online softmax."""
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    # Loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        # Compute QK
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Masking
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        # Online softmax
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        # Load V and update output
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # Update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # Final scaling
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # Store LSE
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    # Store output
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


def flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash attention forward pass that returns both output and LSE.

    Supports GQA (grouped query attention) where num_kv_heads < num_q_heads.

    Args:
        q: Query tensor [batch, seqlen_q, nheads_q, headdim]
        k: Key tensor [batch, seqlen_k, nheads_kv, headdim]
        v: Value tensor [batch, seqlen_k, nheads_kv, headdim]
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        causal: Whether to apply causal masking

    Returns:
        out: Output tensor [batch, seqlen_q, nheads_q, headdim]
        lse: Log-sum-exp tensor [batch, nheads_q, seqlen_q]
    """
    # Ensure contiguous
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    batch, seqlen_q, nheads_q, headdim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape

    assert k.shape == (batch, seqlen_k, nheads_kv, headdim)
    assert v.shape == (batch, seqlen_k, nheads_kv, headdim)
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.dtype == k.dtype == v.dtype

    # Handle GQA by repeating K/V heads
    if nheads_kv != nheads_q:
        assert nheads_q % nheads_kv == 0, f"nheads_q ({nheads_q}) must be divisible by nheads_kv ({nheads_kv})"
        repeat_factor = nheads_q // nheads_kv
        # [batch, seqlen_k, nheads_kv, headdim] -> [batch, seqlen_k, nheads_q, headdim]
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)
        nheads = nheads_q
    else:
        nheads = nheads_q

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK = 128
    num_warps = 4 if headdim <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel_with_lse[grid](
        q,
        k,
        v,
        out,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        seqlen_q // 32,
        seqlen_k // 32,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    # Trim LSE to actual seqlen_q
    lse = lse[:, :, :seqlen_q]

    # Ensure output has same dtype as input
    out = out.to(q.dtype)

    return out, lse


def merge_attention_outputs(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge two attention outputs using online softmax.

    This implements the online softmax merging formula:
    - m_new = max(lse1, lse2)
    - o_new = (exp(lse1 - m_new) * o1 + exp(lse2 - m_new) * o2) / (exp(lse1 - m_new) + exp(lse2 - m_new))
    - lse_new = m_new + log(exp(lse1 - m_new) + exp(lse2 - m_new))

    Args:
        o1: First output [batch, seqlen_q, nheads, headdim]
        lse1: First LSE [batch, nheads, seqlen_q]
        o2: Second output [batch, seqlen_q, nheads, headdim]
        lse2: Second LSE [batch, nheads, seqlen_q]

    Returns:
        o_merged: Merged output [batch, seqlen_q, nheads, headdim]
        lse_merged: Merged LSE [batch, nheads, seqlen_q]
    """
    # lse shape: [batch, nheads, seqlen_q]
    # o shape: [batch, seqlen_q, nheads, headdim]

    # Compute max for numerical stability
    max_lse = torch.maximum(lse1, lse2)

    # Compute scaling factors
    # exp1, exp2 shape: [batch, nheads, seqlen_q]
    exp1 = torch.exp(lse1 - max_lse)
    exp2 = torch.exp(lse2 - max_lse)

    # Reshape for broadcasting with output
    # [batch, nheads, seqlen_q] -> [batch, seqlen_q, nheads, 1]
    exp1_broad = exp1.transpose(1, 2).unsqueeze(-1)
    exp2_broad = exp2.transpose(1, 2).unsqueeze(-1)

    # Merge outputs
    sum_exp = exp1_broad + exp2_broad
    o_merged = (o1 * exp1_broad + o2 * exp2_broad) / sum_exp

    # Compute merged LSE
    lse_merged = max_lse + torch.log(exp1 + exp2)

    # Ensure output has same dtype as input
    o_merged = o_merged.to(o1.dtype)

    return o_merged, lse_merged


def chunked_attention_varlen(
    q: torch.Tensor,
    kv_chunks: List[Tuple[torch.Tensor, torch.Tensor]],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_list: List[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k_list: List[int],
    softmax_scale: Optional[float] = None,
    causal_mask_per_chunk: Optional[List[bool]] = None,
) -> torch.Tensor:
    """
    Compute attention with KV split across multiple chunks.

    This is the core function for chunked prefill. It computes attention
    against each KV chunk and merges results using online softmax.

    For causal attention with chunked KV:
    - First chunk (current tokens): Apply causal mask
    - Previous chunks: No causal mask (all previous tokens are valid context)

    Args:
        q: Query tensor [total_q_tokens, nheads, headdim]
        kv_chunks: List of (K, V) tuples, each [batch, seqlen_k_i, nheads, headdim]
        cu_seqlens_q: Cumulative sequence lengths for Q [batch+1]
        cu_seqlens_k_list: List of cumulative sequence lengths for each KV chunk
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k_list: List of maximum key sequence lengths for each chunk
        softmax_scale: Scaling factor
        causal_mask_per_chunk: Whether to apply causal mask for each chunk

    Returns:
        out: Output tensor [total_q_tokens, nheads, headdim]
    """
    if len(kv_chunks) == 0:
        raise ValueError("Need at least one KV chunk")

    nheads = q.shape[1]
    headdim = q.shape[2]
    batch = cu_seqlens_q.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    if causal_mask_per_chunk is None:
        # Default: causal for last chunk only
        causal_mask_per_chunk = [False] * (len(kv_chunks) - 1) + [True]

    # Initialize accumulated output and LSE
    accumulated_o = None
    accumulated_lse = None

    for chunk_idx, (k_chunk, v_chunk) in enumerate(kv_chunks):
        is_causal = causal_mask_per_chunk[chunk_idx]

        # Reshape Q for batch processing
        # For varlen, we need to handle each sequence separately
        # For simplicity, assume single sequence (batch=1) for now
        q_batched = q.unsqueeze(0)  # [1, total_q, nheads, headdim]

        # Compute attention for this chunk
        chunk_o, chunk_lse = flash_attn_with_lse(
            q_batched,
            k_chunk,
            v_chunk,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )

        # Merge with accumulated
        if accumulated_o is None:
            accumulated_o = chunk_o
            accumulated_lse = chunk_lse
        else:
            accumulated_o, accumulated_lse = merge_attention_outputs(
                accumulated_o, accumulated_lse,
                chunk_o, chunk_lse,
            )

    # Remove batch dimension
    return accumulated_o.squeeze(0)


class ChunkedPrefillState:
    """
    State for tracking chunked prefill progress.

    This class maintains the accumulated attention output and LSE
    across multiple prefill chunks.
    """

    def __init__(self, num_layers: int, dtype: torch.dtype, device: torch.device):
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # Per-layer accumulated outputs
        # Each entry: (accumulated_output, accumulated_lse) or None
        self.layer_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(num_layers)
        ]

        # Track which chunks have been processed
        self.processed_chunks: int = 0

    def update_layer(
        self,
        layer_id: int,
        chunk_output: torch.Tensor,
        chunk_lse: torch.Tensor,
    ):
        """Update accumulated state for a layer with a new chunk's output."""
        if self.layer_states[layer_id] is None:
            self.layer_states[layer_id] = (chunk_output, chunk_lse)
        else:
            acc_o, acc_lse = self.layer_states[layer_id]
            merged_o, merged_lse = merge_attention_outputs(
                acc_o, acc_lse,
                chunk_output, chunk_lse,
            )
            self.layer_states[layer_id] = (merged_o, merged_lse)

    def get_layer_output(self, layer_id: int) -> Optional[torch.Tensor]:
        """Get the final accumulated output for a layer."""
        if self.layer_states[layer_id] is None:
            return None
        return self.layer_states[layer_id][0]

    def clear(self):
        """Clear all accumulated state."""
        self.layer_states = [None for _ in range(self.num_layers)]
        self.processed_chunks = 0


# Test function
def _test_chunked_attention():
    """Test chunked attention correctness against full attention."""
    from flash_attn.flash_attn_interface import flash_attn_func

    torch.manual_seed(42)

    batch, seqlen, nheads, headdim = 1, 1024, 32, 128

    # Generate random Q, K, V
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)

    # Full attention (reference)
    out_ref = flash_attn_func(q, k, v, causal=True)

    # Chunked attention
    chunk_size = 256
    num_chunks = seqlen // chunk_size

    accumulated_o = None
    accumulated_lse = None

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size

        # Q for this chunk
        q_chunk = q[:, start:end, :, :]

        # K, V up to current position (for causal)
        k_context = k[:, :end, :, :]
        v_context = v[:, :end, :, :]

        # Compute attention
        chunk_o, chunk_lse = flash_attn_with_lse(
            q_chunk, k_context, v_context, causal=True
        )

        if accumulated_o is None:
            accumulated_o = chunk_o
            accumulated_lse = chunk_lse
        else:
            # For chunked prefill, we need to concatenate outputs, not merge
            # Because each chunk's Q attends to different K positions
            accumulated_o = torch.cat([accumulated_o, chunk_o], dim=1)

    # Compare
    max_diff = (out_ref - accumulated_o).abs().max().item()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-2, f"Chunked attention differs from reference: {max_diff}"
    print("Test passed!")


if __name__ == "__main__":
    _test_chunked_attention()
