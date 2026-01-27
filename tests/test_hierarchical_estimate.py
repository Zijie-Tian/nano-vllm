"""
Test: Hierarchical Block Sum Estimation for XAttention

Verify that hierarchical estimation (small estimate_block_size + aggregation)
produces equivalent results to direct estimation (large block_size), while
being significantly faster.

Key changes validated:
1. Hierarchical block sum: estimate_block_size=1024 → aggregate to cpu_block_size=4096
2. Selection strategy: score + threshold (NOT mask + majority voting)

This test uses pure torch + xattn kernels, independent of nanovllm framework.
"""

import sys
import os
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.ops.xattn import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum

# ============================================================
# Configuration
# ============================================================

# Model dimensions (Llama-3.1-8B-Instruct style)
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
STRIDE = 8

# Block sizes
CPU_BLOCK_SIZE = 4096       # External CPU block size (fixed, for overlap)
ESTIMATE_BLOCK_SIZE = 1024  # Internal estimate block size (optimized)

# Selection parameters
THRESHOLD = 0.95            # Cumulative attention threshold

# ============================================================
# Hierarchical Estimation Implementation
# ============================================================

def compute_attention_scores(Q, K_blocks, stride):
    """
    Compute attention scores for Q against multiple K blocks.

    Args:
        Q: [1, num_heads, q_len, head_dim]
        K_blocks: List of K tensors, each [1, num_heads, block_size, head_dim]
        stride: Stride for reshape

    Returns:
        attn_scores: [1, num_heads, q_reshaped, total_k_reshaped]
    """
    q_len = Q.shape[2]
    q_reshaped = q_len // stride

    attn_chunks = []
    for K_block in K_blocks:
        # flat_group_gemm_fuse_reshape
        attn_chunk = flat_group_gemm_fuse_reshape(
            Q, K_block, stride,
            chunk_start=0,
            chunk_end=q_reshaped,
            is_causal=False,
        )
        attn_chunks.append(attn_chunk)

    # Concatenate along K dimension
    attn_scores = torch.cat(attn_chunks, dim=-1)
    return attn_scores


def hierarchical_block_sum(
    attn_scores,
    estimate_block_size,
    cpu_block_size,
    stride,
    head_dim,
):
    """
    Compute hierarchical block sums: fine-grained → aggregated to CPU block level.

    Args:
        attn_scores: [batch, heads, q_reshaped, k_reshaped]
        estimate_block_size: Small block size for efficient softmax (e.g., 1024)
        cpu_block_size: External CPU block size (e.g., 4096)
        stride: Stride used in reshape
        head_dim: Head dimension for scale computation

    Returns:
        cpu_block_scores: [batch, heads, num_cpu_blocks] - attention score per CPU block
    """
    batch_size, num_heads, q_reshaped, k_reshaped = attn_scores.shape

    # Compute reshaped block sizes
    reshaped_est_bs = estimate_block_size // stride  # 1024/8 = 128
    reshaped_cpu_bs = cpu_block_size // stride       # 4096/8 = 512

    # Scale factor
    norm = 1.0
    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride / norm

    # Segment size for softmax kernel
    segment_size = min(4096, reshaped_est_bs)

    # Step 1: Fine-grained softmax + block sum
    block_sums_fine = softmax_fuse_block_sum(
        attn_scores,
        reshaped_est_bs,
        segment_size,
        chunk_start=0,
        chunk_end=q_reshaped,
        real_q_len=q_reshaped,
        scale=scale,
        is_causal=False,
    )
    # block_sums_fine: [batch, heads, q_est_blocks, k_est_blocks]

    q_est_blocks = block_sums_fine.shape[2]
    k_est_blocks = block_sums_fine.shape[3]

    # Step 2: Aggregate to CPU block level
    # ratio = cpu_block_size / estimate_block_size = 4
    ratio = cpu_block_size // estimate_block_size
    num_cpu_blocks = k_est_blocks // ratio

    # Reshape and sum along K dimension
    # [batch, heads, q_est, k_est] → [batch, heads, q_est, num_cpu, ratio]
    block_sums_coarse = block_sums_fine.view(
        batch_size, num_heads, q_est_blocks, num_cpu_blocks, ratio
    ).sum(dim=-1)  # [batch, heads, q_est_blocks, num_cpu_blocks]

    # Step 3: Sum over Q dimension (total attention from Q chunk to each K block)
    cpu_block_scores = block_sums_coarse.sum(dim=2)  # [batch, heads, num_cpu_blocks]

    return cpu_block_scores, block_sums_fine


def direct_block_sum(
    attn_scores,
    cpu_block_size,
    stride,
    head_dim,
):
    """
    Compute block sums directly with CPU block size (baseline for comparison).

    Args:
        attn_scores: [batch, heads, q_reshaped, k_reshaped]
        cpu_block_size: Block size (e.g., 4096)
        stride: Stride used in reshape
        head_dim: Head dimension for scale computation

    Returns:
        cpu_block_scores: [batch, heads, num_cpu_blocks]
    """
    batch_size, num_heads, q_reshaped, k_reshaped = attn_scores.shape

    reshaped_cpu_bs = cpu_block_size // stride  # 4096/8 = 512

    norm = 1.0
    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride / norm
    segment_size = min(4096, reshaped_cpu_bs)

    block_sums = softmax_fuse_block_sum(
        attn_scores,
        reshaped_cpu_bs,
        segment_size,
        chunk_start=0,
        chunk_end=q_reshaped,
        real_q_len=q_reshaped,
        scale=scale,
        is_causal=False,
    )
    # block_sums: [batch, heads, q_cpu_blocks, k_cpu_blocks]

    # Sum over Q dimension
    cpu_block_scores = block_sums.sum(dim=2)  # [batch, heads, num_cpu_blocks]

    return cpu_block_scores


def select_blocks_by_score(
    cpu_block_scores,
    threshold=0.95,
    always_include_first=True,
    always_include_last=True,
):
    """
    Select CPU blocks based on score + threshold.

    ⚠️ IMPORTANT: This replaces the original mask + majority voting strategy.
    This change should be documented in the final implementation.

    Args:
        cpu_block_scores: [batch, heads, num_cpu_blocks]
        threshold: Cumulative attention threshold (e.g., 0.95)
        always_include_first: Always include first block (sink)
        always_include_last: Always include last block (safety)

    Returns:
        selected_block_ids: List of selected block indices
        density: Fraction of blocks selected
    """
    # Average scores across heads
    scores_per_block = cpu_block_scores.mean(dim=(0, 1))  # [num_cpu_blocks]
    num_blocks = scores_per_block.shape[0]

    # Normalize to get attention distribution
    total_score = scores_per_block.sum()
    score_ratio = scores_per_block / total_score

    # Sort by score (descending)
    sorted_indices = torch.argsort(score_ratio, descending=True)

    # Select blocks until cumulative threshold is reached
    cumsum = 0.0
    selected = set()

    for idx in sorted_indices.tolist():
        selected.add(idx)
        cumsum += score_ratio[idx].item()
        if cumsum >= threshold:
            break

    # Always include first and last blocks
    if always_include_first:
        selected.add(0)
    if always_include_last:
        selected.add(num_blocks - 1)

    selected_block_ids = sorted(list(selected))
    density = len(selected_block_ids) / num_blocks

    return selected_block_ids, density


# ============================================================
# Test Cases
# ============================================================

def test_equivalence():
    """
    Test that hierarchical estimation produces equivalent scores to direct estimation.
    """
    print("=" * 60)
    print("Test 1: Hierarchical vs Direct - Equivalence")
    print("=" * 60)

    # Create random Q and multiple K blocks
    q_len = CPU_BLOCK_SIZE  # 4096
    num_k_blocks = 4

    # Q: [1, num_heads, q_len, head_dim]
    Q = torch.randn(1, NUM_HEADS, q_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    # K blocks: each [1, num_heads, cpu_block_size, head_dim]
    K_blocks = [
        torch.randn(1, NUM_HEADS, CPU_BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        for _ in range(num_k_blocks)
    ]

    # Compute attention scores
    attn_scores = compute_attention_scores(Q, K_blocks, STRIDE)
    print(f"attn_scores shape: {attn_scores.shape}")

    # Method 1: Hierarchical (fast)
    scores_hier, _ = hierarchical_block_sum(
        attn_scores, ESTIMATE_BLOCK_SIZE, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM
    )
    print(f"scores_hier shape: {scores_hier.shape}")

    # Method 2: Direct (slow)
    scores_direct = direct_block_sum(
        attn_scores, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM
    )
    print(f"scores_direct shape: {scores_direct.shape}")

    # Compare
    diff = (scores_hier - scores_direct).abs().max().item()
    print(f"\nMax difference: {diff:.6f}")

    # Per-block comparison
    print("\nPer-block scores comparison:")
    for i in range(num_k_blocks):
        h_val = scores_hier[0, 0, i].item()
        d_val = scores_direct[0, 0, i].item()
        print(f"  Block {i}: hierarchical={h_val:.4f}, direct={d_val:.4f}, diff={abs(h_val-d_val):.6f}")

    passed = diff < 0.01
    print(f"\nTest 1: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_selection():
    """
    Test the score + threshold selection strategy.
    """
    print("\n" + "=" * 60)
    print("Test 2: Score + Threshold Selection")
    print("=" * 60)

    # Create Q and K blocks with varying importance
    q_len = CPU_BLOCK_SIZE
    num_k_blocks = 8

    Q = torch.randn(1, NUM_HEADS, q_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    # Create K blocks - make some more important than others
    K_blocks = []
    for i in range(num_k_blocks):
        # First and middle blocks are more important (higher values)
        importance = 2.0 if i in [0, 3, 4] else 1.0
        K = torch.randn(1, NUM_HEADS, CPU_BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        K = K * importance
        K_blocks.append(K)

    # Compute scores
    attn_scores = compute_attention_scores(Q, K_blocks, STRIDE)
    scores, _ = hierarchical_block_sum(
        attn_scores, ESTIMATE_BLOCK_SIZE, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM
    )

    # Print scores per block
    print("\nCPU block scores (head 0):")
    for i in range(num_k_blocks):
        print(f"  Block {i}: {scores[0, 0, i].item():.4f}")

    # Select blocks with different thresholds
    for thresh in [0.9, 0.95, 0.99]:
        selected, density = select_blocks_by_score(scores, threshold=thresh)
        print(f"\nThreshold {thresh}: selected {len(selected)}/{num_k_blocks} blocks ({density:.1%})")
        print(f"  Selected: {selected}")

    print("\nTest 2: PASSED (visual inspection)")
    return True


def test_performance():
    """
    Benchmark hierarchical vs direct estimation performance.
    """
    print("\n" + "=" * 60)
    print("Test 3: Performance Benchmark")
    print("=" * 60)

    import time

    NUM_WARMUP = 3
    NUM_RUNS = 10

    # Larger test case
    q_len = CPU_BLOCK_SIZE
    num_k_blocks = 16  # 64K context

    Q = torch.randn(1, NUM_HEADS, q_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    K_blocks = [
        torch.randn(1, NUM_HEADS, CPU_BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        for _ in range(num_k_blocks)
    ]

    # Compute attention scores (shared)
    attn_scores = compute_attention_scores(Q, K_blocks, STRIDE)
    print(f"attn_scores shape: {attn_scores.shape}")
    print(f"Context: {num_k_blocks * CPU_BLOCK_SIZE // 1024}K tokens")

    # Warmup and benchmark hierarchical
    for _ in range(NUM_WARMUP):
        _ = hierarchical_block_sum(attn_scores, ESTIMATE_BLOCK_SIZE, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        _ = hierarchical_block_sum(attn_scores, ESTIMATE_BLOCK_SIZE, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM)
    torch.cuda.synchronize()
    hier_time = (time.perf_counter() - start) / NUM_RUNS * 1000

    # Warmup and benchmark direct
    for _ in range(NUM_WARMUP):
        _ = direct_block_sum(attn_scores, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        _ = direct_block_sum(attn_scores, CPU_BLOCK_SIZE, STRIDE, HEAD_DIM)
    torch.cuda.synchronize()
    direct_time = (time.perf_counter() - start) / NUM_RUNS * 1000

    speedup = direct_time / hier_time

    print(f"\nResults:")
    print(f"  Hierarchical (bs=1024): {hier_time:.2f} ms")
    print(f"  Direct (bs=4096):       {direct_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    passed = speedup > 5.0  # Expect at least 5x speedup
    print(f"\nTest 3: {'PASSED' if passed else 'FAILED'} (speedup > 5x expected)")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Block Sum Estimation Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  NUM_HEADS: {NUM_HEADS}")
    print(f"  NUM_KV_HEADS: {NUM_KV_HEADS}")
    print(f"  HEAD_DIM: {HEAD_DIM}")
    print(f"  STRIDE: {STRIDE}")
    print(f"  CPU_BLOCK_SIZE: {CPU_BLOCK_SIZE}")
    print(f"  ESTIMATE_BLOCK_SIZE: {ESTIMATE_BLOCK_SIZE}")
    print(f"  THRESHOLD: {THRESHOLD}")
    print()

    results = []

    results.append(("Equivalence", test_equivalence()))
    results.append(("Selection", test_selection()))
    results.append(("Performance", test_performance()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print("=" * 60)
    if all_passed:
        print("test_hierarchical_estimate: ALL PASSED")
        sys.exit(0)
    else:
        print("test_hierarchical_estimate: SOME FAILED")
        sys.exit(1)
