"""
Test FlashInfer chunked attention with CPU offload.

Uses single_prefill_with_kv_cache + merge_state for chunked KV processing.
"""

import torch
import flashinfer


# ============================================================
# Core Functions
# ============================================================

def chunked_prefill_causal(q, k_cpu, v_cpu, q_chunk_size, kv_chunk_size):
    """
    Chunked causal attention with KV on CPU.

    q: [seq_q, num_heads, head_dim] on GPU
    k_cpu, v_cpu: [seq_kv, num_kv_heads, head_dim] on CPU
    """
    seq_q = q.shape[0]
    seq_kv = k_cpu.shape[0]
    final_outputs = []

    for q_start in range(0, seq_q, q_chunk_size):
        q_end = min(q_start + q_chunk_size, seq_q)
        q_chunk = q[q_start:q_end]

        merged_output = None
        merged_lse = None

        for kv_start in range(0, seq_kv, kv_chunk_size):
            kv_end = min(kv_start + kv_chunk_size, seq_kv)

            if kv_start >= q_end:
                continue

            k_chunk = k_cpu[kv_start:kv_end].to(q.device, non_blocking=True)
            v_chunk = v_cpu[kv_start:kv_end].to(q.device, non_blocking=True)

            causal = not (kv_end <= q_start)
            partial_out, partial_lse = flashinfer.single_prefill_with_kv_cache(
                q_chunk, k_chunk, v_chunk,
                causal=causal,
                return_lse=True,
            )

            if merged_output is None:
                merged_output, merged_lse = partial_out, partial_lse
            else:
                merged_output, merged_lse = flashinfer.merge_state(
                    merged_output, merged_lse,
                    partial_out, partial_lse,
                )

        final_outputs.append(merged_output)

    return torch.cat(final_outputs, dim=0)


# ============================================================
# Main Test Script
# ============================================================

print("=" * 60)
print("Testing FlashInfer chunked attention with CPU offload")
print("=" * 60)

num_heads = 32
num_kv_heads = 8
head_dim = 128

test_configs = [
    (32768, 8192, 8192),    # 32K tokens
    (65536, 8192, 8192),    # 64K tokens
    (131072, 16384, 16384), # 128K tokens
    # (262144, 16384, 16384), # 256K tokens (slow)
    # (524288, 16384, 16384), # 512K tokens (slow)
]

for seq_len, q_chunk, kv_chunk in test_configs:
    torch.manual_seed(42)

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cpu = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device='cpu')
    v_cpu = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device='cpu')

    # Chunked result
    chunked_out = chunked_prefill_causal(q, k_cpu, v_cpu, q_chunk, kv_chunk)

    # Reference
    k_gpu = k_cpu.to('cuda')
    v_gpu = v_cpu.to('cuda')
    ref_out = flashinfer.single_prefill_with_kv_cache(q, k_gpu, v_gpu, causal=True)

    max_diff = (ref_out - chunked_out).abs().max().item()
    mean_diff = (ref_out - chunked_out).abs().mean().item()

    num_chunks = (seq_len + q_chunk - 1) // q_chunk
    assert max_diff < 1e-2, f"FAILED: max_diff={max_diff:.6f}"
    print(f"seq={seq_len//1024}K, chunks={num_chunks}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

print("\ntest_flashinfer_merge: PASSED")
