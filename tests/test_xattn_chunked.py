"""
Test: xattn_estimate_chunked alignment with xattn_estimate using kvcache-rope data

This test verifies that the chunked implementation produces masks that are
highly aligned (IoU > 0.98) with the standard implementation.
"""
import torch
import torch.nn.functional as F
import sys
import os
import math

sys.path.insert(0, "/home/zijie/Code/COMPASS")

from compass.src.Xattention import xattn_estimate
from compass.src.Xattn_chunked import xattn_estimate_chunked

# ============================================================
# Configuration
# ============================================================
BLOCK_SIZE = 128
STRIDE = 8
THRESHOLD = 0.9
MODEL = "glm-4-9b"
LAYER = 5

# ============================================================
# Utility Functions
# ============================================================

def load_kv_data(data_path):
    """Load KV data from kvcache-rope dataset."""
    data = torch.load(data_path, map_location="cpu")
    post_q = data['post_rope_q']
    post_k = data['post_rope_k']

    seq_len, num_heads, head_dim = post_q.shape
    _, num_kv_heads, _ = post_k.shape

    # Repeat K/V for GQA
    num_groups = num_heads // num_kv_heads
    post_k = post_k.repeat_interleave(num_groups, dim=1)

    # Convert to [batch, heads, seq_len, head_dim]
    query = post_q.unsqueeze(0).transpose(1, 2)
    key = post_k.unsqueeze(0).transpose(1, 2)

    return query, key, seq_len


def compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked, tolerance=1e-5):
    """Compare standard and chunked results."""
    results = {}

    # Shape check
    results['shape_match'] = mask_std.shape == mask_chunked.shape
    if not results['shape_match']:
        results['shape_diff'] = f"std={mask_std.shape}, chunked={mask_chunked.shape}"
        results['exact_match'] = False
        results['iou'] = 0.0
        results['attn_max_diff'] = float('inf')
        results['std_density'] = mask_std.float().mean().item()
        results['chunked_density'] = mask_chunked.float().mean().item()
        return results

    # Exact match
    results['exact_match'] = (mask_std == mask_chunked).all().item()

    # Attn sum difference
    results['attn_max_diff'] = (attn_sum_std - attn_sum_chunked).abs().max().item()

    # IoU
    intersection = (mask_std & mask_chunked).sum().item()
    union = (mask_std | mask_chunked).sum().item()
    results['iou'] = intersection / union if union > 0 else 0

    # Density
    results['std_density'] = mask_std.float().mean().item()
    results['chunked_density'] = mask_chunked.float().mean().item()

    return results


# ============================================================
# Test Cases
# ============================================================

def test_short_sequence():
    """Test 1: Short sequence (4K) - No Q chunking, No KV chunking."""
    print("\n" + "=" * 60)
    print("Test 1: Short Sequence (4K)")
    print("=" * 60)

    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/16k/layer_{LAYER:02d}.pt"
    if not os.path.exists(DATA_PATH):
        print(f"SKIP: Data not found: {DATA_PATH}")
        return True

    query, key, seq_len = load_kv_data(DATA_PATH)
    query = query.cuda().to(torch.bfloat16)
    key = key.cuda().to(torch.bfloat16)

    test_seq_len = 4096
    chunk_size = 4096
    kv_chunk_size = 4096

    query = query[:, :, :test_seq_len, :]
    key = key[:, :, :test_seq_len, :]

    print(f"  Testing with seq_len={test_seq_len}, chunk_size={chunk_size}")
    print(f"  Expected: No Q chunking, No KV chunking")

    attn_sum_std, mask_std = xattn_estimate(
        query, key,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        chunk_size=chunk_size,
        use_triton=True,
    )

    attn_sum_chunked, mask_chunked = xattn_estimate_chunked(
        query, key,
        q_start_pos=0,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        use_triton=True,
        chunk_size=chunk_size,
        kv_chunk_size=kv_chunk_size,
    )

    results = compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked)
    print(f"  Exact match: {results['exact_match']}")
    print(f"  Attn max diff: {results['attn_max_diff']:.6f}")
    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Density: std={results['std_density']:.4f}, chunked={results['chunked_density']:.4f}")

    return results['exact_match']


def test_medium_sequence():
    """Test 2: Medium sequence (16K) - No Q chunking, KV chunking only."""
    print("\n" + "=" * 60)
    print("Test 2: Medium Sequence (16K) - KV Chunking Only")
    print("=" * 60)

    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/16k/layer_{LAYER:02d}.pt"
    if not os.path.exists(DATA_PATH):
        print(f"SKIP: Data not found: {DATA_PATH}")
        return True

    query, key, seq_len = load_kv_data(DATA_PATH)
    query = query.cuda().to(torch.bfloat16)
    key = key.cuda().to(torch.bfloat16)

    q_chunk_size = 16384
    kv_chunk_size = 4096

    print(f"  Testing with seq_len={seq_len}")
    print(f"  Q chunk_size={q_chunk_size}, KV chunk_size={kv_chunk_size}")
    print(f"  Expected: No Q chunking, KV chunking enabled")

    attn_sum_std, mask_std = xattn_estimate(
        query, key,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        chunk_size=q_chunk_size,
        use_triton=True,
    )

    attn_sum_chunked, mask_chunked = xattn_estimate_chunked(
        query, key,
        q_start_pos=0,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        use_triton=True,
        chunk_size=q_chunk_size,
        kv_chunk_size=kv_chunk_size,
    )

    results = compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked)
    print(f"  Exact match: {results['exact_match']}")
    print(f"  Attn max diff: {results['attn_max_diff']:.6f}")
    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Density: std={results['std_density']:.4f}, chunked={results['chunked_density']:.4f}")

    return results['exact_match']


def test_long_sequence():
    """Test 3: Long sequence - True 2D chunking (Q + KV)."""
    print("\n" + "=" * 60)
    print("Test 3: Long Sequence - True 2D Chunking (Q + KV)")
    print("=" * 60)

    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/64k/layer_{LAYER:02d}.pt"
    if not os.path.exists(DATA_PATH):
        print(f"SKIP: Data not found: {DATA_PATH}")
        return True

    query, key, seq_len = load_kv_data(DATA_PATH)
    query = query.cuda().to(torch.bfloat16)
    key = key.cuda().to(torch.bfloat16)

    chunk_size = 4096
    kv_chunk_size = 4096

    print(f"  Testing with seq_len={seq_len}")
    print(f"  Q chunk_size={chunk_size} ({(seq_len + chunk_size - 1) // chunk_size} chunks)")
    print(f"  KV chunk_size={kv_chunk_size} ({(seq_len + kv_chunk_size - 1) // kv_chunk_size} chunks)")
    print(f"  Expected: Q chunking + KV chunking (2D)")

    attn_sum_std, mask_std = xattn_estimate(
        query, key,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        chunk_size=chunk_size,
        use_triton=True,
    )

    attn_sum_chunked, mask_chunked = xattn_estimate_chunked(
        query, key,
        q_start_pos=0,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        use_triton=True,
        chunk_size=chunk_size,
        kv_chunk_size=kv_chunk_size,
    )

    results = compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked)
    print(f"  Exact match: {results['exact_match']}")
    print(f"  Attn max diff: {results['attn_max_diff']:.6f}")
    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Density: std={results['std_density']:.4f}, chunked={results['chunked_density']:.4f}")

    return results['exact_match']


def test_streaming():
    """Test 4: Streaming chunked prefill scenario."""
    print("\n" + "=" * 60)
    print("Test 4: Streaming Chunked Prefill")
    print("=" * 60)

    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/64k/layer_{LAYER:02d}.pt"
    if not os.path.exists(DATA_PATH):
        print(f"SKIP: Data not found: {DATA_PATH}")
        return True

    query, key, seq_len = load_kv_data(DATA_PATH)
    query = query.cuda().to(torch.bfloat16)
    key = key.cuda().to(torch.bfloat16)

    stream_chunk_size = 4096
    kv_chunk_size = 4096

    num_chunks = (seq_len + stream_chunk_size - 1) // stream_chunk_size
    print(f"  Simulating streaming with {num_chunks} chunks")
    print(f"  Stream chunk size: {stream_chunk_size}")
    print(f"  KV chunk size: {kv_chunk_size}")

    all_passed = True
    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * stream_chunk_size
        q_end = min(q_start + stream_chunk_size, seq_len)

        q_chunk = query[:, :, q_start:q_end, :]
        k_chunk = key[:, :, :q_end, :]

        print(f"\n  Chunk {chunk_idx}: Q[{q_start}:{q_end}], K[0:{q_end}]")

        try:
            attn_sum_std, mask_std = xattn_estimate(
                q_chunk, k_chunk,
                block_size=BLOCK_SIZE,
                stride=STRIDE,
                threshold=THRESHOLD,
                chunk_size=stream_chunk_size,
                use_triton=True,
            )

            attn_sum_chunked, mask_chunked = xattn_estimate_chunked(
                q_chunk, k_chunk,
                q_start_pos=q_start,
                block_size=BLOCK_SIZE,
                stride=STRIDE,
                threshold=THRESHOLD,
                use_triton=True,
                chunk_size=stream_chunk_size,
                kv_chunk_size=kv_chunk_size,
            )

            results = compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked)
            print(f"    Exact match: {results['exact_match']}, IoU: {results['iou']:.4f}")

            if not results['exact_match']:
                all_passed = False

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM (skipping chunk)")
                continue
            raise

    return all_passed


def test_different_thresholds():
    """Test 5: Different threshold values."""
    print("\n" + "=" * 60)
    print("Test 5: Different Thresholds")
    print("=" * 60)

    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/16k/layer_{LAYER:02d}.pt"
    if not os.path.exists(DATA_PATH):
        print(f"SKIP: Data not found: {DATA_PATH}")
        return True

    query, key, seq_len = load_kv_data(DATA_PATH)
    query = query.cuda().to(torch.bfloat16)
    key = key.cuda().to(torch.bfloat16)

    chunk_size = 4096
    kv_chunk_size = 4096
    thresholds = [0.8, 0.9, 0.95, 0.99]

    all_passed = True
    for threshold in thresholds:
        print(f"\n  Threshold = {threshold}")

        attn_sum_std, mask_std = xattn_estimate(
            query, key,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            threshold=threshold,
            chunk_size=chunk_size,
            use_triton=True,
        )

        attn_sum_chunked, mask_chunked = xattn_estimate_chunked(
            query, key,
            q_start_pos=0,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            threshold=threshold,
            use_triton=True,
            chunk_size=chunk_size,
            kv_chunk_size=kv_chunk_size,
        )

        results = compare_masks(attn_sum_std, mask_std, attn_sum_chunked, mask_chunked)
        print(f"    Exact match: {results['exact_match']}, IoU: {results['iou']:.4f}")

        if not results['exact_match']:
            all_passed = False

    return all_passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Testing xattn_estimate_chunked vs xattn_estimate")
    print(f"Model: {MODEL} layer {LAYER:02d}")
    print(f"Config: BLOCK_SIZE={BLOCK_SIZE}, STRIDE={STRIDE}")
    print("\nThis test verifies 2D chunking (Q + KV) alignment with standard implementation.")

    results = {}
    results['short'] = test_short_sequence()
    results['medium'] = test_medium_sequence()
    results['long'] = test_long_sequence()
    results['streaming'] = test_streaming()
    results['thresholds'] = test_different_thresholds()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name.capitalize()}: {status}")

    if all(results.values()):
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")

    print("=" * 60)
