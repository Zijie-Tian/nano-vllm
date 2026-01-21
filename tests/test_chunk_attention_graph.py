#!/usr/bin/env python3
"""
Test: Pre-allocated chunk pair graphs for block sparse attention.

Each (Q_chunk, K_chunk) pair has its own captured CUDA graph.
Zero copy_() during replay - all data pre-filled.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_chunk_attention_graph.py
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from nanovllm.ops.chunked_attention import flash_attn_with_lse, merge_attention_outputs


@dataclass
class ChunkAttentionGraph:
    """Container for a captured chunk attention graph."""
    graph: torch.cuda.CUDAGraph
    static_q: torch.Tensor
    static_k: torch.Tensor
    static_v: torch.Tensor
    static_output: torch.Tensor
    static_lse: torch.Tensor
    causal: bool


def capture_chunk_attention_graph(
    chunk_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool = False,
) -> ChunkAttentionGraph:
    """Capture a CUDA graph for single chunk attention."""
    static_q = torch.zeros(1, chunk_size, num_heads, head_dim, dtype=dtype, device=device)
    static_k = torch.zeros(1, chunk_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    static_v = torch.zeros(1, chunk_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    static_q.normal_()
    static_k.normal_()
    static_v.normal_()

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = flash_attn_with_lse(static_q, static_k, static_v, scale, causal)
        torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            static_output, static_lse = flash_attn_with_lse(static_q, static_k, static_v, scale, causal)

    torch.cuda.synchronize()

    return ChunkAttentionGraph(
        graph=graph,
        static_q=static_q,
        static_k=static_k,
        static_v=static_v,
        static_output=static_output,
        static_lse=static_lse,
        causal=causal,
    )


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    chunk_size = 64
    num_chunks = 4
    num_heads = 8
    num_kv_heads = 8
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)
    seq_len = chunk_size * num_chunks

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Chunk size: {chunk_size}, Num chunks: {num_chunks}")
    print(f"Total graphs: {num_chunks * (num_chunks + 1) // 2}")

    # Test data
    full_q = torch.randn(1, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    full_k = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    full_v = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Reference
    with torch.inference_mode():
        full_output, _ = flash_attn_with_lse(full_q, full_k, full_v, scale, causal=True)

    # Capture all graphs
    graphs: List[List[Optional[ChunkAttentionGraph]]] = [[None] * num_chunks for _ in range(num_chunks)]
    for q_idx in range(num_chunks):
        for k_idx in range(q_idx + 1):
            graphs[q_idx][k_idx] = capture_chunk_attention_graph(
                chunk_size, num_heads, num_kv_heads, head_dim, scale, device, dtype,
                causal=(k_idx == q_idx)
            )
    print("All graphs captured")

    # Pre-fill static tensors
    for q_idx in range(num_chunks):
        for k_idx in range(q_idx + 1):
            g = graphs[q_idx][k_idx]
            g.static_q.copy_(full_q[:, q_idx*chunk_size:(q_idx+1)*chunk_size])
            g.static_k.copy_(full_k[:, k_idx*chunk_size:(k_idx+1)*chunk_size])
            g.static_v.copy_(full_v[:, k_idx*chunk_size:(k_idx+1)*chunk_size])
    print("Static tensors pre-filled")

    # Replay and merge
    chunked_output = torch.zeros_like(full_output)
    for q_idx in range(num_chunks):
        acc_out, acc_lse = None, None
        for k_idx in range(q_idx + 1):
            g = graphs[q_idx][k_idx]
            g.graph.replay()
            out, lse = g.static_output.clone(), g.static_lse.clone()
            if acc_out is None:
                acc_out, acc_lse = out, lse
            else:
                with torch.inference_mode():
                    acc_out, acc_lse = merge_attention_outputs(acc_out, acc_lse, out, lse)
        chunked_output[:, q_idx*chunk_size:(q_idx+1)*chunk_size] = acc_out

    torch.cuda.synchronize()

    # Compare
    all_pass = True
    for q_idx in range(num_chunks):
        s, e = q_idx * chunk_size, (q_idx + 1) * chunk_size
        diff = (full_output[:, s:e] - chunked_output[:, s:e]).abs().max().item()
        status = "✅" if diff < 1e-2 else "❌"
        print(f"Q[{q_idx}]: max_diff={diff:.2e} {status}")
        if diff >= 1e-2:
            all_pass = False

    print("✅ PASSED" if all_pass else "❌ FAILED")


if __name__ == "__main__":
    main()
