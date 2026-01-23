#!/usr/bin/env python3
"""
Test: Reuse a single CUDA Graph across all layers and all chunk pairs.

Key insight: LLM layers have identical computation structure.
We only need 2 graphs (causal + non-causal), reused for all (layer, Q_i, K_j) combinations.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_chunk_attention_graph_reuse.py
"""

from dataclasses import dataclass

import torch

from nanovllm.ops.chunked_attention import flash_attn_with_lse, merge_attention_outputs


@dataclass
class ReusableChunkGraph:
    """A single graph that can be reused with copy_() updates."""
    graph: torch.cuda.CUDAGraph
    static_q: torch.Tensor
    static_k: torch.Tensor
    static_v: torch.Tensor
    static_output: torch.Tensor
    static_lse: torch.Tensor


def capture_reusable_graph(
    chunk_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
) -> ReusableChunkGraph:
    """Capture ONE graph to be reused for all chunk pairs."""
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

    return ReusableChunkGraph(
        graph=graph,
        static_q=static_q,
        static_k=static_k,
        static_v=static_v,
        static_output=static_output,
        static_lse=static_lse,
    )


def replay_with_copy(graph: ReusableChunkGraph, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Replay graph after updating static tensors with copy_()."""
    graph.static_q.copy_(q)
    graph.static_k.copy_(k)
    graph.static_v.copy_(v)
    graph.graph.replay()
    return graph.static_output.clone(), graph.static_lse.clone()


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    chunk_size = 64
    num_chunks = 4
    num_layers = 3  # Simulate multiple layers
    num_heads = 8
    num_kv_heads = 8
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)
    seq_len = chunk_size * num_chunks

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Chunk size: {chunk_size}, Num chunks: {num_chunks}, Num layers: {num_layers}")
    print(f"Only 2 graphs (causal + non-causal) for ALL layer × chunk combinations")

    # Capture only 2 graphs
    graph_causal = capture_reusable_graph(
        chunk_size, num_heads, num_kv_heads, head_dim, scale, device, dtype, causal=True
    )
    graph_non_causal = capture_reusable_graph(
        chunk_size, num_heads, num_kv_heads, head_dim, scale, device, dtype, causal=False
    )
    print("2 graphs captured (causal + non-causal)")

    all_pass = True

    for layer_id in range(num_layers):
        # Different Q/K/V for each layer (simulating different layer outputs)
        full_q = torch.randn(1, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        full_k = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        full_v = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Reference: full causal attention
        with torch.inference_mode():
            full_output, _ = flash_attn_with_lse(full_q, full_k, full_v, scale, causal=True)

        # Chunked with graph reuse
        chunked_output = torch.zeros_like(full_output)

        for q_idx in range(num_chunks):
            q_chunk = full_q[:, q_idx*chunk_size:(q_idx+1)*chunk_size]
            acc_out, acc_lse = None, None

            for k_idx in range(q_idx + 1):
                k_chunk = full_k[:, k_idx*chunk_size:(k_idx+1)*chunk_size]
                v_chunk = full_v[:, k_idx*chunk_size:(k_idx+1)*chunk_size]

                # Reuse graph with copy_()
                graph = graph_causal if k_idx == q_idx else graph_non_causal
                out, lse = replay_with_copy(graph, q_chunk, k_chunk, v_chunk)

                if acc_out is None:
                    acc_out, acc_lse = out, lse
                else:
                    with torch.inference_mode():
                        acc_out, acc_lse = merge_attention_outputs(acc_out, acc_lse, out, lse)

            chunked_output[:, q_idx*chunk_size:(q_idx+1)*chunk_size] = acc_out

        torch.cuda.synchronize()

        # Compare
        max_diff = (full_output - chunked_output).abs().max().item()
        status = "✅" if max_diff < 1e-2 else "❌"
        print(f"Layer {layer_id}: max_diff={max_diff:.2e} {status}")
        if max_diff >= 1e-2:
            all_pass = False

    print("✅ PASSED - Single graph reuse across layers works!" if all_pass else "❌ FAILED")


if __name__ == "__main__":
    main()
