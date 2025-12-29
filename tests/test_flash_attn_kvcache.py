"""
Test script for flash_attn_with_kvcache based chunked prefill.

Verifies that chunked prefill produces identical results to full attention.
"""

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache


def chunk_prefill(q_full, k_full, v_full, k_cache, v_cache, cache_seqlens, chunk_size):
    """
    Chunked prefill using flash_attn_with_kvcache.

    Args:
        q_full, k_full, v_full: [batch, total_seq_len, heads, head_dim]
        k_cache, v_cache: [batch, max_seq_len, kv_heads, head_dim]
        cache_seqlens: [batch] - current cache lengths
        chunk_size: size of each chunk

    Returns:
        output: [batch, total_seq_len, heads, head_dim]
    """
    total_len = q_full.shape[1]
    outputs = []

    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)

        q_chunk = q_full[:, start:end]
        k_chunk = k_full[:, start:end]
        v_chunk = v_full[:, start:end]

        out = flash_attn_with_kvcache(
            q_chunk,
            k_cache,
            v_cache,
            k=k_chunk,
            v=v_chunk,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        outputs.append(out)

        cache_seqlens += (end - start)

    return torch.cat(outputs, dim=1)


def reference_attention(q, k, v):
    """Standard flash attention as reference."""
    return flash_attn_func(q, k, v, causal=True)


def test_chunked_prefill_correctness():
    """Test that chunked prefill matches full attention."""

    batch_size = 1
    num_heads = 32
    num_kv_heads = 8  # GQA
    head_dim = 128
    max_seq_len = 131072  # 128K

    test_configs = [
        (1024, 256),     # 1K tokens, 256 chunk
        (2048, 512),     # 2K tokens, 512 chunk
        (4096, 1024),    # 4K tokens, 1K chunk
        (4096, 2048),    # 4K tokens, 2K chunk (2 chunks)
        (8192, 2048),    # 8K tokens, 2K chunk (4 chunks)
        (16384, 4096),   # 16K tokens, 4K chunk
        (32768, 4096),   # 32K tokens, 4K chunk
        (65536, 8192),   # 64K tokens, 8K chunk
        (131072, 8192),  # 128K tokens, 8K chunk (16 chunks)
    ]

    for seq_len, chunk_size in test_configs:
        print(f"\nTesting seq_len={seq_len}, chunk_size={chunk_size}...")

        # Generate random input
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                        dtype=torch.float16, device='cuda')
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                        dtype=torch.float16, device='cuda')
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                        dtype=torch.float16, device='cuda')

        # Expand K/V for non-GQA reference
        k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=2)
        v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=2)

        # Reference: full attention
        ref_out = reference_attention(q, k_expanded, v_expanded)

        # Chunked prefill with KV cache
        k_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                              dtype=torch.float16, device='cuda')
        v_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                              dtype=torch.float16, device='cuda')
        cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')

        chunked_out = chunk_prefill(q, k, v, k_cache, v_cache, cache_seqlens, chunk_size)

        # Compare
        max_diff = (ref_out - chunked_out).abs().max().item()
        mean_diff = (ref_out - chunked_out).abs().mean().item()

        # Verify cache was filled correctly
        assert cache_seqlens[0].item() == seq_len, f"Cache seqlen mismatch: {cache_seqlens[0].item()} != {seq_len}"

        # Check K/V cache content
        k_cache_diff = (k_cache[:, :seq_len] - k).abs().max().item()
        v_cache_diff = (v_cache[:, :seq_len] - v).abs().max().item()

        print(f"  Output max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")
        print(f"  KV cache diff: k={k_cache_diff:.6f}, v={v_cache_diff:.6f}")

        # Tolerance for fp16
        tolerance = 1e-2
        if max_diff < tolerance:
            print(f"  PASSED")
        else:
            print(f"  FAILED (max_diff {max_diff:.6f} >= {tolerance})")
            return False

    return True


def test_incremental_decode():
    """Test that decode after chunked prefill works correctly."""

    batch_size = 1
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 8192

    prefill_len = 2048
    chunk_size = 512
    num_decode_steps = 10

    print(f"\nTesting incremental decode after chunked prefill...")
    print(f"  Prefill: {prefill_len} tokens, chunk_size={chunk_size}")
    print(f"  Decode: {num_decode_steps} steps")

    torch.manual_seed(42)

    # Prefill phase
    q_prefill = torch.randn(batch_size, prefill_len, num_heads, head_dim,
                            dtype=torch.float16, device='cuda')
    k_prefill = torch.randn(batch_size, prefill_len, num_kv_heads, head_dim,
                            dtype=torch.float16, device='cuda')
    v_prefill = torch.randn(batch_size, prefill_len, num_kv_heads, head_dim,
                            dtype=torch.float16, device='cuda')

    k_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                          dtype=torch.float16, device='cuda')
    v_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                          dtype=torch.float16, device='cuda')
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')

    # Run chunked prefill
    prefill_out = chunk_prefill(q_prefill, k_prefill, v_prefill,
                                 k_cache, v_cache, cache_seqlens, chunk_size)

    print(f"  After prefill: cache_seqlens={cache_seqlens[0].item()}")

    # Decode phase - one token at a time
    for step in range(num_decode_steps):
        q_decode = torch.randn(batch_size, 1, num_heads, head_dim,
                               dtype=torch.float16, device='cuda')
        k_decode = torch.randn(batch_size, 1, num_kv_heads, head_dim,
                               dtype=torch.float16, device='cuda')
        v_decode = torch.randn(batch_size, 1, num_kv_heads, head_dim,
                               dtype=torch.float16, device='cuda')

        decode_out = flash_attn_with_kvcache(
            q_decode,
            k_cache,
            v_cache,
            k=k_decode,
            v=v_decode,
            cache_seqlens=cache_seqlens,
            causal=True,
        )

        cache_seqlens += 1

        assert decode_out.shape == (batch_size, 1, num_heads, head_dim)

    expected_len = prefill_len + num_decode_steps
    actual_len = cache_seqlens[0].item()

    print(f"  After decode: cache_seqlens={actual_len}")

    if actual_len == expected_len:
        print(f"  PASSED")
        return True
    else:
        print(f"  FAILED: expected {expected_len}, got {actual_len}")
        return False


def test_batch_processing():
    """Test chunked prefill with batch > 1."""

    batch_size = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 4096
    seq_len = 2048
    chunk_size = 512

    print(f"\nTesting batch processing (batch_size={batch_size})...")

    torch.manual_seed(42)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                    dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                    dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                    dtype=torch.float16, device='cuda')

    k_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                          dtype=torch.float16, device='cuda')
    v_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                          dtype=torch.float16, device='cuda')
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')

    out = chunk_prefill(q, k, v, k_cache, v_cache, cache_seqlens, chunk_size)

    # Verify all batches have correct cache length
    assert (cache_seqlens == seq_len).all(), f"Cache seqlens mismatch: {cache_seqlens}"
    assert out.shape == (batch_size, seq_len, num_heads, head_dim)

    # Compare with reference for each batch item
    k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=2)
    v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=2)
    ref_out = reference_attention(q, k_expanded, v_expanded)

    max_diff = (ref_out - out).abs().max().item()

    print(f"  Output shape: {out.shape}")
    print(f"  Max diff vs reference: {max_diff:.6f}")

    if max_diff < 1e-2:
        print(f"  PASSED")
        return True
    else:
        print(f"  FAILED")
        return False


# ============================================================
# Main Test Script
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing flash_attn_with_kvcache chunked prefill")
    print("=" * 60)

    all_passed = True

    all_passed &= test_chunked_prefill_correctness()
    all_passed &= test_incremental_decode()
    all_passed &= test_batch_processing()

    print("\n" + "=" * 60)
    if all_passed:
        print("test_flash_attn_kvcache: ALL TESTS PASSED")
    else:
        print("test_flash_attn_kvcache: SOME TESTS FAILED")
    print("=" * 60)
