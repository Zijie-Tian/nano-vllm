"""
Test script for chunked attention correctness.

Validates that chunked prefill using flash_attn_with_lse + merge_attention_outputs
produces the same result as full flash_attn_varlen_func.

Scenario: Simulating chunked prefill where we process query chunk by chunk.
For each query chunk i:
- KV contains all tokens from chunk 0 to chunk i
- Previous KV chunks (0 to i-1): full attention (no causal mask)
- Current KV chunk (i): causal attention (diagonal block)
"""

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs

# ============================================================
# Utility Functions
# ============================================================

def compute_chunked_prefill_for_chunk(
    q_chunk: torch.Tensor,
    kv_chunks: list,
    current_chunk_idx: int,
) -> torch.Tensor:
    """
    Compute attention for a single query chunk against all KV chunks up to current.

    This simulates chunked prefill for query chunk `current_chunk_idx`:
    - KV chunks 0 to current_chunk_idx-1: full attention (all previous tokens visible)
    - KV chunk current_chunk_idx: causal attention (diagonal block)

    Args:
        q_chunk: [batch, chunk_size, nheads, headdim] - current query chunk
        kv_chunks: List of (k, v) tuples, each [batch, chunk_size, nheads, headdim]
        current_chunk_idx: Index of the current chunk being processed

    Returns:
        out: [batch, chunk_size, nheads, headdim]
    """
    accumulated_o = None
    accumulated_lse = None

    for i in range(current_chunk_idx + 1):
        k_chunk, v_chunk = kv_chunks[i]

        # Previous chunks: no causal mask (all tokens visible)
        # Current chunk (diagonal): causal mask
        is_diagonal = (i == current_chunk_idx)

        chunk_o, chunk_lse = flash_attn_with_lse(
            q_chunk, k_chunk, v_chunk, causal=is_diagonal
        )

        if accumulated_o is None:
            accumulated_o = chunk_o
            accumulated_lse = chunk_lse
        else:
            accumulated_o, accumulated_lse = merge_attention_outputs(
                accumulated_o, accumulated_lse,
                chunk_o, chunk_lse
            )

    return accumulated_o


def compute_reference_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute reference causal attention using flash_attn_func.

    Args:
        q, k, v: [batch, seqlen, nheads, headdim]

    Returns:
        out: [batch, seqlen, nheads, headdim]
    """
    return flash_attn_func(q, k, v, causal=True)


# ============================================================
# Main Test Script
# ============================================================

torch.manual_seed(42)

# Test configurations: (batch, num_chunks, chunk_size, nheads, headdim)
TEST_CASES = [
    (1, 4, 256, 8, 128),
    (1, 4, 512, 8, 128),
    (1, 8, 512, 8, 128),
    (1, 4, 1024, 8, 128),
    (1, 4, 1024, 32, 128),   # More heads
    (1, 8, 256, 8, 64),      # Smaller head dim
]

DTYPES = [torch.float16, torch.bfloat16]

print("=" * 80)
print("Test: Chunked Prefill Attention vs Reference (flash_attn_func causal)")
print("=" * 80)
print("Simulating chunked prefill: Q chunk attends to all KV chunks up to current")
print("  - Previous KV chunks: full attention (no causal mask)")
print("  - Current KV chunk (diagonal): causal attention")
print()

all_passed = True

for dtype in DTYPES:
    print(f"--- dtype: {dtype} ---")

    for batch, num_chunks, chunk_size, nheads, headdim in TEST_CASES:
        seqlen = num_chunks * chunk_size

        # Generate full Q, K, V
        q_full = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
        k_full = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
        v_full = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)

        # Reference: full causal attention
        out_ref = compute_reference_causal(q_full, k_full, v_full)

        # Split into chunks
        q_chunks = [q_full[:, i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
        kv_chunks = [
            (k_full[:, i*chunk_size:(i+1)*chunk_size],
             v_full[:, i*chunk_size:(i+1)*chunk_size])
            for i in range(num_chunks)
        ]

        # Compute chunked prefill for each query chunk
        out_chunks = []
        for chunk_idx in range(num_chunks):
            chunk_out = compute_chunked_prefill_for_chunk(
                q_chunks[chunk_idx],
                kv_chunks,
                chunk_idx,
            )
            out_chunks.append(chunk_out)

        # Concatenate chunked outputs
        out_chunked = torch.cat(out_chunks, dim=1)

        # Compare
        diff = (out_ref - out_chunked).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Tolerance: fp16/bf16 have limited precision
        tol = 1e-2
        passed = max_diff < tol
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] seqlen={seqlen:5d} chunks={num_chunks} "
            f"chunk_size={chunk_size:4d} heads={nheads:2d} dim={headdim:3d} "
            f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}"
        )

    print()

print("=" * 80)
assert all_passed, "Some tests failed!"
print("test_chunked_attention: PASSED")
