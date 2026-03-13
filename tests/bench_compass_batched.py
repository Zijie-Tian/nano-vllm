"""
Benchmark: Batched TMAC qGEMM vs Micro-call overhead.

Validates the performance improvement of COMPASS Level 1 + Level 2 batching:
- Level 1: Batch sub-blocks (M = N_subs * 256, N=1) vs micro-calls (M=256, N=1) * N_subs
- Level 2: Batch KV heads  (M = N_subs * 256, N=8) vs Level 1 * 8

Uses COMPASS's actual parameters:
  bits=2, g=4, K=128 (head_dim), group_size=128, act_group_size=64, zero_point=True
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm.ops.tvm_qgemm.qgemm import (
    QGeMMLUTBitsCodegen,
    QGeMMLUTBitsPreprocessorCodegen,
)
import tvm

# COMPASS parameters
BITS = 2
G = 4
K = 128          # head_dim
GROUP_SIZE = 128
ACT_GROUP_SIZE = 64
FINE_GRAIN = 128  # tokens per sub-block
BM = 256          # = FINE_GRAIN * BITS
SUBS_PER_BLOCK = 32  # 4096 / 128
NUM_KV_HEADS = 8
NUM_Q_GROUPS = 32
NGE = 8 // G  # 2


def compile_qgemm(M, N, label=""):
    """Compile a single qGEMM function for given M, N."""
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
    codegen = QGeMMLUTBitsCodegen(
        dtype="int8", target=target,
        name=f"build/bench_batched_{label}",
        tune=False, verify=False, num_threads=32,
        bits=BITS, g=G, group_size=GROUP_SIZE,
        act_group_size=ACT_GROUP_SIZE, out_dtype="float32",
        m_groups=-1, zero_point=True,
    )
    func, _ = codegen.compile(M, N, K)
    return func, codegen


def compile_preproc(N):
    """Compile preprocessor for given N."""
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
    preproc = QGeMMLUTBitsPreprocessorCodegen(
        dtype="int8", target=target,
        name=f"build/bench_batched_preproc_n{N}",
        tune=False, verify=False, num_threads=32,
        g=G, act_group_size=ACT_GROUP_SIZE,
        out_dtype="float32", bits=BITS,
    )
    func, _ = preproc.compile(N, K)
    return func, preproc


def make_arrays(M, N, codegen, dev):
    """Create input/output TVM arrays for qGEMM."""
    bm = codegen.bm
    A_tvm = tvm.nd.array(
        np.random.randint(0, 256, size=(M // bm, K // G, bm // NGE), dtype="uint8"), dev)
    LUT_tvm = tvm.nd.array(
        np.random.normal(size=(N, K // G, 1 << G)).astype("int8"), dev)
    Scales_tvm = tvm.nd.array(
        np.random.normal(size=(M // bm, K // GROUP_SIZE, bm // BITS * 2)).astype("float32"), dev)
    LUT_Scales_tvm = tvm.nd.array(
        np.random.normal(size=(N, K // ACT_GROUP_SIZE)).astype("float32"), dev)
    LUT_Biases_tvm = tvm.nd.array(
        np.random.normal(size=(N, K // ACT_GROUP_SIZE)).astype("float32"), dev)
    C_tvm = tvm.nd.array(
        np.zeros((N, M // BITS), dtype="float32"), dev)
    return A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm


def bench_micro_calls(func_small, codegen_small, dev, num_subs, num_warmup=3):
    """Benchmark: call small qGEMM num_subs times (current COMPASS pattern)."""
    M_small = BM  # 256
    arrays = make_arrays(M_small, 1, codegen_small, dev)

    # Warmup
    for _ in range(num_warmup):
        func_small(*arrays)

    start = time.time()
    for _ in range(num_subs):
        func_small(*arrays)
    return time.time() - start


def bench_batched(func_big, codegen_big, dev, M_big, N, num_warmup=3):
    """Benchmark: single batched qGEMM call."""
    arrays = make_arrays(M_big, N, codegen_big, dev)

    # Warmup
    for _ in range(num_warmup):
        func_big(*arrays)

    start = time.time()
    func_big(*arrays)
    return time.time() - start


def bench_full_loop_micro(func_small, codegen_small, func_preproc_small, preproc_small,
                          dev, num_blocks, num_warmup=2):
    """Simulate full COMPASS loop with micro-calls: q_groups × kv_heads × subs."""
    M_small = BM
    num_subs = num_blocks * SUBS_PER_BLOCK
    total_qgemm = NUM_Q_GROUPS * NUM_KV_HEADS * num_subs
    total_preproc = NUM_Q_GROUPS * NUM_KV_HEADS

    # Pre-allocate
    q_arrays = make_arrays(M_small, 1, codegen_small, dev)
    B_tvm = tvm.nd.array(np.random.normal(size=(1, K)).astype("float32"), dev)
    LUT_S = tvm.nd.array(np.zeros((1, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    LUT_B = tvm.nd.array(np.zeros((1, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    QLUT = tvm.nd.array(np.zeros((1, K // G, 1 << G), dtype="int8"), dev)

    # Warmup
    for _ in range(num_warmup):
        func_preproc_small(B_tvm, LUT_S, LUT_B, QLUT)
        func_small(*q_arrays)

    start = time.time()
    for _ in range(NUM_Q_GROUPS):
        for _ in range(NUM_KV_HEADS):
            func_preproc_small(B_tvm, LUT_S, LUT_B, QLUT)
            for _ in range(num_subs):
                func_small(*q_arrays)
    elapsed = time.time() - start

    return elapsed, total_qgemm, total_preproc


def bench_full_loop_L1(func_big, codegen_big, func_preproc_small, preproc_small,
                       dev, num_blocks, num_warmup=2):
    """Simulate Level 1 batching: q_groups × kv_heads × 1 batched call."""
    num_subs = num_blocks * SUBS_PER_BLOCK
    M_big = num_subs * BM

    q_arrays = make_arrays(M_big, 1, codegen_big, dev)
    B_tvm = tvm.nd.array(np.random.normal(size=(1, K)).astype("float32"), dev)
    LUT_S = tvm.nd.array(np.zeros((1, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    LUT_B = tvm.nd.array(np.zeros((1, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    QLUT = tvm.nd.array(np.zeros((1, K // G, 1 << G), dtype="int8"), dev)

    # Warmup
    for _ in range(num_warmup):
        func_preproc_small(B_tvm, LUT_S, LUT_B, QLUT)
        func_big(*q_arrays)

    start = time.time()
    for _ in range(NUM_Q_GROUPS):
        for _ in range(NUM_KV_HEADS):
            func_preproc_small(B_tvm, LUT_S, LUT_B, QLUT)
            func_big(*q_arrays)
    elapsed = time.time() - start

    total_calls = NUM_Q_GROUPS * NUM_KV_HEADS  # qGEMM calls
    return elapsed, total_calls


def bench_full_loop_L2(func_big_n8, codegen_big_n8, func_preproc_n8, preproc_n8,
                       dev, num_blocks, num_warmup=2):
    """Simulate Level 2 batching: q_groups × 1 batched call (N=8)."""
    num_subs = num_blocks * SUBS_PER_BLOCK
    M_big = num_subs * BM
    N = NUM_KV_HEADS  # 8

    q_arrays = make_arrays(M_big, N, codegen_big_n8, dev)
    B_tvm = tvm.nd.array(np.random.normal(size=(N, K)).astype("float32"), dev)
    LUT_S = tvm.nd.array(np.zeros((N, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    LUT_B = tvm.nd.array(np.zeros((N, K // ACT_GROUP_SIZE), dtype="float32"), dev)
    QLUT = tvm.nd.array(np.zeros((N, K // G, 1 << G), dtype="int8"), dev)

    # Warmup
    for _ in range(num_warmup):
        func_preproc_n8(B_tvm, LUT_S, LUT_B, QLUT)
        func_big_n8(*q_arrays)

    start = time.time()
    for _ in range(NUM_Q_GROUPS):
        func_preproc_n8(B_tvm, LUT_S, LUT_B, QLUT)
        func_big_n8(*q_arrays)
    elapsed = time.time() - start

    total_calls = NUM_Q_GROUPS  # qGEMM calls
    return elapsed, total_calls


if __name__ == "__main__":
    dev = tvm.cpu(0)

    # ================================================================
    # Compile all needed functions
    # ================================================================
    print("Compiling TMAC operators...")

    # Small (current COMPASS): M=256, N=1
    print("  [1/6] qGEMM small (M=256, N=1)")
    func_small, cg_small = compile_qgemm(BM, 1, "small")

    # Preprocessor N=1
    print("  [2/6] Preprocessor (N=1)")
    func_pp1, pp1 = compile_preproc(1)

    # Level 1 batched: M varies, N=1
    # Compile for block counts: 1-7 (32K) + 8,16,32 (longer contexts)
    block_counts = list(range(1, 8)) + [8, 16, 32]
    funcs_L1 = {}
    for nb in block_counts:
        M_big = nb * SUBS_PER_BLOCK * BM
        print(f"  [3/6] qGEMM L1 (M={M_big}, N=1) for {nb} blocks ({nb*4}K tokens)")
        f, c = compile_qgemm(M_big, 1, f"L1_b{nb}")
        funcs_L1[nb] = (f, c)

    # Level 2 batched: M varies, N=8
    funcs_L2 = {}
    for nb in block_counts:
        M_big = nb * SUBS_PER_BLOCK * BM
        print(f"  [4/6] qGEMM L2 (M={M_big}, N=8) for {nb} blocks ({nb*4}K tokens)")
        f, c = compile_qgemm(M_big, NUM_KV_HEADS, f"L2_b{nb}")
        funcs_L2[nb] = (f, c)

    # Preprocessor N=8
    print("  [5/6] Preprocessor (N=8)")
    func_pp8, pp8 = compile_preproc(NUM_KV_HEADS)

    print("  Done.\n")

    # ================================================================
    # Benchmark
    # ================================================================
    print("=" * 70)
    print("COMPASS TMAC Batching Benchmark")
    print("  K=128, bits=2, g=4, 8 KV heads, 32 Q groups")
    print("=" * 70)

    header = f"{'Blocks':>6} | {'Micro (curr)':>14} | {'L1 Batched':>14} | {'L1 Speedup':>10} | {'L2 Batched':>14} | {'L2 Speedup':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for nb in block_counts:
        # Current: micro-calls
        t_micro, n_qgemm, n_pp = bench_full_loop_micro(
            func_small, cg_small, func_pp1, pp1, dev, nb)

        # Level 1: batch sub-blocks
        func_L1, cg_L1 = funcs_L1[nb]
        t_L1, n_L1 = bench_full_loop_L1(
            func_L1, cg_L1, func_pp1, pp1, dev, nb)

        # Level 2: batch sub-blocks + KV heads
        func_L2, cg_L2 = funcs_L2[nb]
        t_L2, n_L2 = bench_full_loop_L2(
            func_L2, cg_L2, func_pp8, pp8, dev, nb)

        speedup_L1 = t_micro / t_L1 if t_L1 > 0 else float('inf')
        speedup_L2 = t_micro / t_L2 if t_L2 > 0 else float('inf')

        label = f"{nb*4}K"
        print(f"{nb:>6} ({label:>5}) | {t_micro:>11.3f}s   | {t_L1:>11.3f}s   | {speedup_L1:>8.1f}x  | {t_L2:>11.3f}s   | {speedup_L2:>8.1f}x")

    # Estimate for 32K (chunks 1-7, 32 layers) and 128K (chunks 1-31, 32 layers)
    for ctx_label, max_chunks in [("32K", 7), ("128K", 31)]:
        print(f"\n{'=' * 70}")
        print(f"{ctx_label} Full Sequence Estimate (32 layers, chunks 1-{max_chunks})")
        print(f"{'=' * 70}")

        total_micro = 0.0
        total_L1 = 0.0
        total_L2 = 0.0

        for chunk in range(1, max_chunks + 1):
            nb = chunk
            t_m, _, _ = bench_full_loop_micro(func_small, cg_small, func_pp1, pp1, dev, nb)
            # Find best matching compiled function
            best_L1_key = min(funcs_L1.keys(), key=lambda x: abs(x - nb) if x >= nb else 9999)
            if best_L1_key < nb:
                best_L1_key = max(funcs_L1.keys())
            best_L2_key = min(funcs_L2.keys(), key=lambda x: abs(x - nb) if x >= nb else 9999)
            if best_L2_key < nb:
                best_L2_key = max(funcs_L2.keys())
            func_L1, cg_L1 = funcs_L1[best_L1_key]
            func_L2, cg_L2 = funcs_L2[best_L2_key]
            t_l1, _ = bench_full_loop_L1(func_L1, cg_L1, func_pp1, pp1, dev, nb)
            t_l2, _ = bench_full_loop_L2(func_L2, cg_L2, func_pp8, pp8, dev, nb)
            total_micro += t_m
            total_L1 += t_l1
            total_L2 += t_l2

        # Scale by 32 layers
        total_micro *= 32
        total_L1 *= 32
        total_L2 *= 32

        print(f"\n  {'Method':<20} {'Total Time':>12} {'vs BLASST (6.7s)':>16}")
        print(f"  {'-'*50}")
        print(f"  {'Micro (current)':<20} {total_micro:>10.1f}s   {total_micro/6.7:>12.1f}x slower")
        print(f"  {'L1 (batch subs)':<20} {total_L1:>10.1f}s   {total_L1/6.7:>12.1f}x slower")
        print(f"  {'L2 (batch heads)':<20} {total_L2:>10.1f}s   {total_L2/6.7:>12.1f}x slower")
        print(f"\n  L1 speedup over current: {total_micro/total_L1:.1f}x")
        print(f"  L2 speedup over current: {total_micro/total_L2:.1f}x")
        print(f"  BLASST total:            6.7s (GPU Triton, for reference)")
