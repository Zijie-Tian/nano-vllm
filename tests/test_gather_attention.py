"""
Standalone mini-test for sub-block gather + packed attention correctness.

Verifies that:
1. gather_subblocks_to_staging produces correct data
2. Packed attention (sub-blocks from different blocks concatenated) gives same
   result as full-block-by-block attention + LSE merge

No model required. Only needs torch + flash_attn.

Usage:
    python tests/test_gather_attention.py
"""

import torch
import sys
import math


# ============================================================
# Helpers
# ============================================================

def flash_attn_with_lse(q, k, v, softmax_scale, causal=False):
    """Compute attention + return output and LSE using flash_attn.

    q: [1, seq_q, heads, dim]
    k: [1, seq_k, heads, dim]
    v: [1, seq_k, heads, dim]

    Returns: (out, lse)
      out: [1, seq_q, heads, dim]
      lse: [1, heads, seq_q]
    """
    from flash_attn import flash_attn_func
    # flash_attn_func expects [batch, seqlen, nheads, headdim]
    o, softmax_lse, *_ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    # softmax_lse: [batch, nheads, seqlen_q]
    return o, softmax_lse


def merge_attention_outputs(o1, lse1, o2, lse2):
    """Merge two attention outputs using log-sum-exp.

    o1, o2: [1, seq, heads, dim]
    lse1, lse2: [1, heads, seq]

    For each position, the merged output is:
      w1 = exp(lse1 - max) / (exp(lse1 - max) + exp(lse2 - max))
      w2 = 1 - w1
      merged_o = w1 * o1 + w2 * o2
    """
    # lse: [1, heads, seq] → [1, seq, heads, 1]
    lse1_t = lse1.transpose(1, 2).unsqueeze(-1)  # [1, seq, heads, 1]
    lse2_t = lse2.transpose(1, 2).unsqueeze(-1)

    max_lse = torch.maximum(lse1_t, lse2_t)
    exp1 = torch.exp(lse1_t - max_lse)
    exp2 = torch.exp(lse2_t - max_lse)
    sum_exp = exp1 + exp2

    w1 = exp1 / sum_exp
    w2 = exp2 / sum_exp

    merged_o = w1 * o1 + w2 * o2
    merged_lse = (max_lse + torch.log(sum_exp)).squeeze(-1).transpose(1, 2)
    return merged_o, merged_lse


# ============================================================
# Test 1: gather_subblocks correctness
# ============================================================

def test_gather_subblocks():
    """Verify that CPU staging buffer gather produces correct data."""
    print("test_gather_subblocks...", end=" ")

    num_layers = 2
    num_blocks = 4
    block_size = 512  # small for testing
    kv_heads = 4
    head_dim = 64
    sub_block_size = 128
    dtype = torch.float16

    # Simulate CPU KV cache: [layers, blocks, block_size, heads, dim]
    k_cache_cpu = torch.randn(
        num_layers, num_blocks, block_size, kv_heads, head_dim,
        dtype=dtype, device="cpu"
    )

    # Staging buffer
    staging_k = torch.zeros(block_size, kv_heads, head_dim, dtype=dtype, device="cpu")

    # Select sub-blocks: block 0 → [1, 3], block 2 → [0]
    selections = [
        (0, [1, 3]),
        (2, [0]),
    ]
    layer_id = 0

    # Gather
    offset = 0
    for cpu_block_id, sub_indices in selections:
        for si in sub_indices:
            src_start = si * sub_block_size
            src_end = src_start + sub_block_size
            staging_k[offset:offset + sub_block_size].copy_(
                k_cache_cpu[layer_id, cpu_block_id, src_start:src_end]
            )
            offset += sub_block_size

    assert offset == 3 * sub_block_size  # 3 sub-blocks gathered
    # Verify: staging[0:128] should match k_cache_cpu[0, 0, 128:256]
    assert torch.equal(staging_k[0:128], k_cache_cpu[0, 0, 128:256])
    # staging[128:256] should match k_cache_cpu[0, 0, 384:512]
    assert torch.equal(staging_k[128:256], k_cache_cpu[0, 0, 384:512])
    # staging[256:384] should match k_cache_cpu[0, 2, 0:128]
    assert torch.equal(staging_k[256:384], k_cache_cpu[0, 2, 0:128])

    print("PASSED")


# ============================================================
# Test 2: Packed attention vs per-block attention equivalence
# ============================================================

def test_packed_attention_equivalence():
    """
    Verify packed attention gives same result as per-block attention + merge.

    Scenario:
    - 3 historical "blocks", each 256 tokens
    - Compare:
      A) Per-block: attend to each block separately, merge with LSE
      B) Packed: concatenate all blocks, single attention call
    """
    print("test_packed_attention_equivalence...", end=" ")

    torch.manual_seed(42)
    seq_q = 256    # query length
    block_size = 256
    num_heads = 8
    head_dim = 64
    softmax_scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(1, seq_q, num_heads, head_dim, dtype=dtype, device=device)

    blocks_k = [
        torch.randn(1, block_size, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(3)
    ]
    blocks_v = [
        torch.randn(1, block_size, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(3)
    ]

    # ---- Path A: Full per-block attention + merge ----
    o_acc, lse_acc = None, None
    for i in range(3):
        o_i, lse_i = flash_attn_with_lse(q, blocks_k[i], blocks_v[i], softmax_scale)
        if o_acc is None:
            o_acc, lse_acc = o_i, lse_i
        else:
            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o_i, lse_i)
    result_per_block = o_acc

    # ---- Path B: Concatenated single attention ----
    all_k = torch.cat(blocks_k, dim=1)
    all_v = torch.cat(blocks_v, dim=1)
    result_packed, _ = flash_attn_with_lse(q, all_k, all_v, softmax_scale)

    max_diff = (result_per_block - result_packed).abs().max().item()
    assert max_diff < 0.02, f"Max diff too large: {max_diff}"

    print(f"PASSED (max_diff={max_diff:.6f})")


# ============================================================
# Test 3: Partial sub-block selection (sparse) equivalence
# ============================================================

def test_sparse_packed_attention():
    """
    Sparse selection: pack selected sub-blocks → single attention
    should equal attending to each sub-block individually → merge.
    """
    print("test_sparse_packed_attention...", end=" ")

    torch.manual_seed(123)
    seq_q = 128
    block_size = 256
    sub_block_size = 128
    num_heads = 8
    head_dim = 64
    softmax_scale = 1.0 / (head_dim ** 0.5)
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(1, seq_q, num_heads, head_dim, dtype=dtype, device=device)

    blocks_k = [
        torch.randn(1, block_size, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(2)
    ]
    blocks_v = [
        torch.randn(1, block_size, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(2)
    ]

    # Select: block 0 sub-block 1, block 1 sub-block 0
    selections = [(0, [1]), (1, [0])]

    # ---- Path A: Per-sub-block attention + merge ----
    o_acc, lse_acc = None, None
    for bid, sub_indices in selections:
        for si in sub_indices:
            start = si * sub_block_size
            end = start + sub_block_size
            k_sub = blocks_k[bid][:, start:end, :, :]
            v_sub = blocks_v[bid][:, start:end, :, :]
            o_i, lse_i = flash_attn_with_lse(q, k_sub, v_sub, softmax_scale)
            if o_acc is None:
                o_acc, lse_acc = o_i, lse_i
            else:
                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o_i, lse_i)
    result_individual = o_acc

    # ---- Path B: Pack selected sub-blocks into one, single attention ----
    packed_k = []
    packed_v = []
    for bid, sub_indices in selections:
        for si in sub_indices:
            start = si * sub_block_size
            end = start + sub_block_size
            packed_k.append(blocks_k[bid][:, start:end, :, :])
            packed_v.append(blocks_v[bid][:, start:end, :, :])
    packed_k = torch.cat(packed_k, dim=1)
    packed_v = torch.cat(packed_v, dim=1)
    result_packed, _ = flash_attn_with_lse(q, packed_k, packed_v, softmax_scale)

    max_diff = (result_individual - result_packed).abs().max().item()
    assert max_diff < 0.02, f"Max diff too large: {max_diff}"

    print(f"PASSED (max_diff={max_diff:.6f})")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_gather_subblocks()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests.")
        sys.exit(0)

    test_packed_attention_equivalence()
    test_sparse_packed_attention()

    print("\nAll tests PASSED.")
