"""
Test: Compare xattn_estimate vs xattn_estimate_chunked
Verify that chunked estimation with EXTERNAL chunking produces the same mask as standard estimation.
"""

import sys
import os
import torch
import warnings

sys.path.insert(0, "/home/zijie/Code/COMPASS")

from compass.src.Xattention import xattn_estimate
from compass.src.Xattn_chunked import xattn_estimate_chunked

# ============================================================
# Configuration
# ============================================================

BLOCK_SIZE = 64
STRIDE = 4
THRESHOLD = 0.9
CHUNK_SIZE = 4096

# ============================================================
# Utility Functions
# ============================================================

def load_qkv(path):
    """Load saved QKV data."""
    data = torch.load(path, map_location="cpu")
    print(f"Loaded: {path}")
    print(f"  Query shape: {data['query'].shape}")
    print(f"  Key shape: {data['key'].shape}")
    print(f"  Layer: {data['layer_id']}, Density: {data['density']:.2%}")
    return data

def compare_masks(mask1, mask2, name1="standard", name2="chunked"):
    """Compare two masks and report differences."""
    if mask1.shape != mask2.shape:
        print(f"Shape mismatch: {name1}={mask1.shape}, {name2}={mask2.shape}")
        return False

    diff = (mask1 != mask2).sum().item()
    total = mask1.numel()
    match_rate = (total - diff) / total * 100

    print(f"  Match rate: {match_rate:.4f}% ({total - diff}/{total})")

    if diff > 0:
        diff_indices = torch.where(mask1 != mask2)
        print(f"  First 5 diff positions: {list(zip(*[idx[:5].tolist() for idx in diff_indices]))}")

    return diff == 0


def run_chunked_externally(query, key, q_start_pos, block_size, stride, threshold, chunk_size):
    """
    Run xattn_estimate_chunked with EXTERNAL chunking.
    This simulates how chunked prefill should be used in practice.
    """
    batch_size, num_heads, q_len, head_dim = query.shape
    _, _, k_len, _ = key.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # If Q fits in one chunk, call directly
    if q_len <= chunk_size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xattn_estimate_chunked(
                query, key,
                q_start_pos=q_start_pos,
                block_size=block_size,
                stride=stride,
                threshold=threshold,
                use_triton=True,
                chunk_size=chunk_size,
            )

    # External chunking: split Q and call for each chunk
    num_q_chunks = (q_len + chunk_size - 1) // chunk_size
    print(f"    External chunking: {num_q_chunks} chunks")

    combined_attn_sum = torch.zeros(
        batch_size, num_heads, q_block_num, k_block_num,
        dtype=query.dtype, device=query.device
    )
    combined_mask = torch.zeros(
        batch_size, num_heads, q_block_num, k_block_num,
        dtype=torch.bool, device=query.device
    )

    q_block_offset = 0
    for q_chunk_idx in range(num_q_chunks):
        q_chunk_start = q_chunk_idx * chunk_size
        q_chunk_end = min((q_chunk_idx + 1) * chunk_size, q_len)

        q_chunk = query[:, :, q_chunk_start:q_chunk_end, :]

        # For causal attention, K accumulates up to current Q position
        k_end = q_start_pos + q_chunk_end
        k_chunk = key[:, :, :k_end, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attn_sum_chunk, mask_chunk = xattn_estimate_chunked(
                q_chunk, k_chunk,
                q_start_pos=q_start_pos + q_chunk_start,
                block_size=block_size,
                stride=stride,
                threshold=threshold,
                use_triton=True,
                chunk_size=chunk_size,
            )

        # Place chunk results into combined output
        chunk_q_blocks = mask_chunk.shape[2]
        chunk_k_blocks = mask_chunk.shape[3]
        combined_attn_sum[:, :, q_block_offset:q_block_offset+chunk_q_blocks, :chunk_k_blocks] = attn_sum_chunk
        combined_mask[:, :, q_block_offset:q_block_offset+chunk_q_blocks, :chunk_k_blocks] = mask_chunk
        q_block_offset += chunk_q_blocks

    return combined_attn_sum, combined_mask


def test_single_qkv(qkv_path):
    """Test a single QKV file."""
    data = load_qkv(qkv_path)
    query = data["query"].cuda().to(torch.bfloat16)
    key = data["key"].cuda().to(torch.bfloat16)

    seq_len = query.shape[2]
    print(f"\nTesting with seq_len={seq_len}")
    print("=" * 60)

    # Run standard xattn_estimate
    print("[1] Running standard xattn_estimate...")
    try:
        attn_sum_std, mask_std = xattn_estimate(
            query, key,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            threshold=THRESHOLD,
            chunk_size=CHUNK_SIZE,
            use_triton=True,
        )
        print(f"  mask shape: {mask_std.shape}, density: {mask_std.float().mean().item():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run chunked xattn_estimate with EXTERNAL chunking
    print("[2] Running chunked xattn_estimate (external chunking)...")
    try:
        attn_sum_chunked, mask_chunked = run_chunked_externally(
            query, key,
            q_start_pos=0,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            threshold=THRESHOLD,
            chunk_size=CHUNK_SIZE,
        )
        print(f"  mask shape: {mask_chunked.shape}, density: {mask_chunked.float().mean().item():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare results
    print("[3] Comparing results...")
    chunked_q_blocks = mask_chunked.shape[2]
    chunked_k_blocks = mask_chunked.shape[3]

    # Extract comparable region from standard mask
    mask_std_comparable = mask_std[:, :, :chunked_q_blocks, :chunked_k_blocks]

    # Compare masks
    masks_match = compare_masks(mask_std_comparable, mask_chunked, "standard", "chunked")

    # Compare attn_sums
    attn_sum_std_comparable = attn_sum_std[:, :, :chunked_q_blocks, :chunked_k_blocks]
    if attn_sum_std_comparable.shape == attn_sum_chunked.shape:
        attn_diff = (attn_sum_std_comparable - attn_sum_chunked).abs().max().item()
        print(f"  Attn sum max diff: {attn_diff:.6f}")
    else:
        print(f"  Attn sum shape mismatch")

    # Clean up GPU memory
    del query, key, attn_sum_std, mask_std, attn_sum_chunked, mask_chunked
    torch.cuda.empty_cache()

    return masks_match

# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":
    qkv_files = [
        "/home/zijie/Code/COMPASS/results/kvcache/qkv_3688.pt",   # 4K
        "/home/zijie/Code/COMPASS/results/kvcache/qkv_7888.pt",   # 8K
        "/home/zijie/Code/COMPASS/results/kvcache/qkv_15685.pt",  # 16K
        "/home/zijie/Code/COMPASS/results/kvcache/qkv_32485.pt",  # 32K
        "/home/zijie/Code/COMPASS/results/kvcache/qkv_64891.pt",  # 64K
    ]

    # Find all available files
    available_files = [p for p in qkv_files if os.path.exists(p)]

    if not available_files:
        print("No QKV file found. Run xattn with XATTN_SAVE_KV=1 first.")
        sys.exit(1)

    print(f"Found {len(available_files)} QKV files to test")
    print(f"Testing EXTERNAL chunking (chunk_size={CHUNK_SIZE})")

    all_passed = True
    results = []

    for qkv_path in available_files:
        passed = test_single_qkv(qkv_path)
        seq_len = int(os.path.basename(qkv_path).replace("qkv_", "").replace(".pt", ""))
        results.append((seq_len, passed))
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for seq_len, passed in results:
        status = "PASSED" if passed else "FAILED"
        chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  seq_len={seq_len} ({chunks} chunk{'s' if chunks > 1 else ''}): {status}")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
