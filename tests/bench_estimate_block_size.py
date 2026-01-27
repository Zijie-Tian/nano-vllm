"""
Benchmark: block_size impact on XAttention estimate phase performance.

This script tests how different block_size values affect the performance of:
1. flat_group_gemm_fuse_reshape (estimate GEMM)
2. softmax_fuse_block_sum (estimate softmax + block aggregation)

Key insight: The current select_blocks uses global kvcache_block_size for estimation,
which may not be optimal for the Triton kernels.
"""

import sys
import os
import torch
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.ops.xattn import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum

# ============================================================
# Configuration
# ============================================================

# Test configurations
BLOCK_SIZES = [64, 128, 256, 512]  # BSA optimal is 128
STRIDE = 8
NUM_WARMUP = 3
NUM_RUNS = 10

# Model dimensions (Llama-3.1-8B-Instruct)
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Context lengths to test
CONTEXT_LENGTHS = [16384, 32768, 65536]  # 16K, 32K, 64K

# ============================================================
# Benchmark Functions
# ============================================================

def benchmark_flat_group_gemm(Q, K, stride, block_size, num_warmup=3, num_runs=10):
    """
    Benchmark flat_group_gemm_fuse_reshape kernel.

    Args:
        Q: [batch, heads, q_len, head_dim]
        K: [batch, heads, k_len, head_dim]
        stride: Stride for reshape
        block_size: Block size (affects alignment requirements)

    Returns:
        (avg_time_ms, output_tensor)
    """
    q_len = Q.shape[2]
    k_len = K.shape[2]

    # Compute reshaped dimensions
    reshaped_q_len = q_len // stride
    reshaped_k_len = k_len // stride
    reshaped_block_size = block_size // stride

    # Warmup
    for _ in range(num_warmup):
        _ = flat_group_gemm_fuse_reshape(
            Q, K, stride,
            chunk_start=0,
            chunk_end=reshaped_q_len,
            is_causal=False,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        output = flat_group_gemm_fuse_reshape(
            Q, K, stride,
            chunk_start=0,
            chunk_end=reshaped_q_len,
            is_causal=False,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms, output


def benchmark_softmax_fuse_block_sum(attn_weights, reshaped_block_size, num_warmup=3, num_runs=10):
    """
    Benchmark softmax_fuse_block_sum kernel.

    Args:
        attn_weights: [batch, heads, q_len, k_len] attention weights
        reshaped_block_size: Block size in reshaped space

    Returns:
        avg_time_ms
    """
    batch_size, num_heads, q_len, k_len = attn_weights.shape
    head_dim = HEAD_DIM
    stride = STRIDE
    norm = 1.0

    # segment_size must divide k_len and be >= reshaped_block_size
    segment_size = min(4096, reshaped_block_size)

    # Ensure k_len is divisible by segment_size
    if k_len % segment_size != 0:
        # Pad k_len
        pad_size = segment_size - (k_len % segment_size)
        attn_weights = torch.nn.functional.pad(attn_weights, (0, pad_size), value=0)
        k_len = attn_weights.shape[3]

    # Scale factor
    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride / norm

    # Warmup
    for _ in range(num_warmup):
        _ = softmax_fuse_block_sum(
            attn_weights,
            reshaped_block_size,
            segment_size,
            chunk_start=0,
            chunk_end=q_len,
            real_q_len=q_len,
            scale=scale,
            is_causal=False,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        output = softmax_fuse_block_sum(
            attn_weights,
            reshaped_block_size,
            segment_size,
            chunk_start=0,
            chunk_end=q_len,
            real_q_len=q_len,
            scale=scale,
            is_causal=False,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def run_estimate_benchmark(q_len, k_len, block_size, stride=STRIDE):
    """
    Run full estimate benchmark for given configuration.

    Args:
        q_len: Query length
        k_len: Key length (usually same as q_len for current chunk scenario)
        block_size: Block size to test
        stride: Stride for reshape

    Returns:
        dict with timing results
    """
    # Create random Q and K tensors
    # Shape: [batch, heads, seq_len, head_dim]
    Q = torch.randn(1, NUM_HEADS, q_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, NUM_HEADS, k_len, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    reshaped_block_size = block_size // stride
    reshaped_q_len = q_len // stride
    reshaped_k_len = k_len // stride

    # Benchmark GEMM
    gemm_time, attn_weights = benchmark_flat_group_gemm(
        Q, K, stride, block_size,
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS
    )

    # Benchmark softmax + block sum
    softmax_time = benchmark_softmax_fuse_block_sum(
        attn_weights, reshaped_block_size,
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS
    )

    # Clean up
    del Q, K, attn_weights
    torch.cuda.empty_cache()

    return {
        "q_len": q_len,
        "k_len": k_len,
        "block_size": block_size,
        "reshaped_block_size": reshaped_block_size,
        "gemm_time_ms": gemm_time,
        "softmax_time_ms": softmax_time,
        "total_time_ms": gemm_time + softmax_time,
    }


# ============================================================
# Main Benchmark
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark block_size impact on estimate phase")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--ctx-len", type=int, default=None,
                        help="Single context length to test (default: test multiple)")
    args = parser.parse_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)
    device_name = torch.cuda.get_device_name(args.gpu)
    print(f"Using GPU {args.gpu}: {device_name}")
    print()

    # Determine context lengths to test
    if args.ctx_len:
        context_lengths = [args.ctx_len]
    else:
        context_lengths = CONTEXT_LENGTHS

    print("=" * 80)
    print("Benchmark: block_size impact on XAttention estimate phase")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  NUM_HEADS: {NUM_HEADS}")
    print(f"  NUM_KV_HEADS: {NUM_KV_HEADS}")
    print(f"  HEAD_DIM: {HEAD_DIM}")
    print(f"  STRIDE: {STRIDE}")
    print(f"  BLOCK_SIZES: {BLOCK_SIZES}")
    print(f"  NUM_WARMUP: {NUM_WARMUP}")
    print(f"  NUM_RUNS: {NUM_RUNS}")
    print()

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*80}")
        print(f"Context Length: {ctx_len // 1024}K ({ctx_len} tokens)")
        print(f"{'='*80}")

        # Pad to alignment
        alignment = STRIDE * 128  # Triton BLOCK_M requirement
        padded_len = ((ctx_len + alignment - 1) // alignment) * alignment
        print(f"Padded to: {padded_len} tokens (alignment={alignment})")
        print()

        results = []
        for block_size in BLOCK_SIZES:
            print(f"Testing block_size={block_size} (reshaped={block_size // STRIDE})...", end=" ")
            try:
                result = run_estimate_benchmark(padded_len, padded_len, block_size)
                results.append(result)
                print(f"GEMM={result['gemm_time_ms']:.2f}ms, "
                      f"Softmax={result['softmax_time_ms']:.2f}ms, "
                      f"Total={result['total_time_ms']:.2f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if results:
            all_results.extend(results)

            # Print summary table for this context length
            print(f"\n--- Summary for {ctx_len // 1024}K context ---")
            print(f"{'block_size':>12} {'reshaped':>10} {'GEMM (ms)':>12} {'Softmax (ms)':>14} {'Total (ms)':>12} {'Speedup':>10}")
            print("-" * 74)

            baseline_total = results[0]["total_time_ms"]
            for r in results:
                speedup = baseline_total / r["total_time_ms"]
                print(f"{r['block_size']:>12} {r['reshaped_block_size']:>10} "
                      f"{r['gemm_time_ms']:>12.2f} {r['softmax_time_ms']:>14.2f} "
                      f"{r['total_time_ms']:>12.2f} {speedup:>9.2f}x")

    # Final summary across all context lengths
    if len(context_lengths) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"{'ctx_len':>10} {'block_size':>12} {'GEMM (ms)':>12} {'Softmax (ms)':>14} {'Total (ms)':>12}")
        print("-" * 64)
        for r in all_results:
            print(f"{r['q_len']//1024:>9}K {r['block_size']:>12} "
                  f"{r['gemm_time_ms']:>12.2f} {r['softmax_time_ms']:>14.2f} "
                  f"{r['total_time_ms']:>12.2f}")

    # Find optimal block_size for softmax
    print(f"\n{'='*80}")
    print("ANALYSIS: Optimal block_size for softmax_fuse_block_sum")
    print(f"{'='*80}")

    for ctx_len in context_lengths:
        ctx_results = [r for r in all_results if r["q_len"] == ((ctx_len + STRIDE * 128 - 1) // (STRIDE * 128)) * (STRIDE * 128)]
        if ctx_results:
            best = min(ctx_results, key=lambda x: x["softmax_time_ms"])
            worst = max(ctx_results, key=lambda x: x["softmax_time_ms"])
            improvement = worst["softmax_time_ms"] / best["softmax_time_ms"]
            print(f"Context {ctx_len // 1024}K:")
            print(f"  Best:  block_size={best['block_size']} ({best['softmax_time_ms']:.2f}ms)")
            print(f"  Worst: block_size={worst['block_size']} ({worst['softmax_time_ms']:.2f}ms)")
            print(f"  Potential improvement: {improvement:.2f}x")

    print("\nbench_estimate_block_size: DONE")


if __name__ == "__main__":
    main()
