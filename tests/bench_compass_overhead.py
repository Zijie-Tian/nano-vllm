"""
Benchmark: COMPASS TMAC micro-call overhead analysis.

Measures the cost of calling many tiny TVM qGEMM operations (as COMPASS does)
versus what a single batched call would cost.

COMPASS select_blocks inner loop (per chunk, per layer):
  for q_grp in range(num_q_groups):        # 32 (4096 / 128)
    for kv_h in range(num_kv_heads):        # 8
      preprocess(q_group)                   # 1 preprocessor call
      for sub_block in range(num_sub):      # 32 * num_blocks
        qgemm(sub_block, qlut)              # 1 qgemm call

This benchmark replicates these exact parameters.
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


def compile_compass_operators():
    """Compile TMAC operators with COMPASS's exact parameters."""
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
    bits = 2
    K = 128   # head_dim
    N = 1     # single KV head
    M = 128 * bits  # 256 = FINE_GRAIN * bits

    print(f"Compiling TMAC operators (M={M}, N={N}, K={K}, bits={bits})...")

    # Preprocessor
    preproc = QGeMMLUTBitsPreprocessorCodegen(
        dtype="int8", target=target, name="bench_compass_preproc",
        tune=False, verify=False, num_threads=1, g=4,
        act_group_size=64, out_dtype="float32", bits=bits,
    )
    func_preproc, _ = preproc.compile(N, K)

    # QGEMM
    codegen = QGeMMLUTBitsCodegen(
        dtype="int8", target=target, name="bench_compass_qgemm",
        tune=False, verify=False, num_threads=1, bits=bits, g=4,
        group_size=128, act_group_size=64, out_dtype="float32",
        m_groups=-1, zero_point=True,
    )
    func_qgemm, _ = codegen.compile(M, N, K)

    return func_preproc, func_qgemm, preproc, codegen


def bench_single_call(func_qgemm, codegen, dev, num_warmup=5, num_runs=100):
    """Benchmark a single tiny qGEMM call latency."""
    bits = 2
    K = 128
    N = 1
    M = 256
    g = 4
    bm = codegen.bm
    nge = 8 // g

    A_shape = (M // bm, K // g, bm // nge)
    LUT_shape = (N, K // g, 1 << g)
    Scales_shape = (M // bm, K // codegen.group_size, bm // bits * 2)  # zero_point=True
    LUT_Scales_shape = (N, K // codegen.act_group_size)
    LUT_Biases_shape = (N, K // codegen.act_group_size)
    C_shape = (N, M // bits)

    A_tvm = tvm.nd.array(np.random.randint(0, 256, size=A_shape, dtype="uint8"), dev)
    LUT_tvm = tvm.nd.array(np.random.normal(size=LUT_shape).astype("int8"), dev)
    Scales_tvm = tvm.nd.array(np.random.normal(size=Scales_shape).astype("float32"), dev)
    LUT_Scales_tvm = tvm.nd.array(np.random.normal(size=LUT_Scales_shape).astype("float32"), dev)
    LUT_Biases_tvm = tvm.nd.array(np.random.normal(size=LUT_Biases_shape).astype("float32"), dev)
    C_tvm = tvm.nd.array(np.zeros(C_shape, dtype="float32"), dev)

    # Warmup
    for _ in range(num_warmup):
        func_qgemm(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)

    # Measure
    start = time.time()
    for _ in range(num_runs):
        func_qgemm(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)
    elapsed = time.time() - start

    return elapsed / num_runs


def bench_single_preproc(func_preproc, preproc, dev, num_warmup=5, num_runs=100):
    """Benchmark a single preprocessor call latency."""
    K = 128
    N = 1
    g = 4

    B_tvm = tvm.nd.array(np.random.normal(size=(N, K)).astype("float32"), dev)
    LUT_Scales_tvm = tvm.nd.array(np.zeros((N, K // preproc.act_group_size), dtype="float32"), dev)
    LUT_Biases_tvm = tvm.nd.array(np.zeros((N, K // preproc.act_group_size), dtype="float32"), dev)
    QLUT_tvm = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype="int8"), dev)

    for _ in range(num_warmup):
        func_preproc(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)

    start = time.time()
    for _ in range(num_runs):
        func_preproc(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
    elapsed = time.time() - start

    return elapsed / num_runs


def bench_compass_loop(func_preproc, func_qgemm, preproc, codegen, dev,
                       num_blocks, num_kv_heads=8, num_q_groups=32):
    """
    Simulate COMPASS select_blocks inner loop for one layer, one chunk.

    Total calls per layer:
      preprocessor: num_q_groups * num_kv_heads
      qgemm: num_q_groups * num_kv_heads * (num_blocks * 32)
    """
    bits = 2
    K = 128
    N = 1
    M = 256
    g = 4
    bm = codegen.bm
    nge = 8 // g
    fine_grain = 128
    subs_per_block = 4096 // fine_grain  # 32

    num_sub_blocks = num_blocks * subs_per_block
    total_preproc = num_q_groups * num_kv_heads
    total_qgemm = num_q_groups * num_kv_heads * num_sub_blocks

    # Pre-allocate arrays
    A_shape = (M // bm, K // g, bm // nge)
    LUT_shape = (N, K // g, 1 << g)
    Scales_shape = (M // bm, K // codegen.group_size, bm // bits * 2)  # zero_point=True
    LUT_Scales_shape = (N, K // codegen.act_group_size)
    LUT_Biases_shape = (N, K // codegen.act_group_size)
    C_shape = (N, M // bits)

    A_tvm = tvm.nd.array(np.random.randint(0, 256, size=A_shape, dtype="uint8"), dev)
    LUT_tvm = tvm.nd.array(np.random.normal(size=LUT_shape).astype("int8"), dev)
    Scales_tvm = tvm.nd.array(np.random.normal(size=Scales_shape).astype("float32"), dev)
    LUT_Scales_tvm = tvm.nd.array(np.random.normal(size=LUT_Scales_shape).astype("float32"), dev)
    LUT_Biases_tvm = tvm.nd.array(np.random.normal(size=LUT_Biases_shape).astype("float32"), dev)
    C_tvm = tvm.nd.array(np.zeros(C_shape, dtype="float32"), dev)

    B_tvm = tvm.nd.array(np.random.normal(size=(N, K)).astype("float32"), dev)
    QLUT_tvm = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype="int8"), dev)

    print(f"\n  Simulating COMPASS loop: {num_blocks} blocks, "
          f"{num_kv_heads} KV heads, {num_q_groups} Q groups")
    print(f"  Total preprocessor calls: {total_preproc}")
    print(f"  Total qGEMM calls:        {total_qgemm}")

    # --- Simulate the full loop ---
    start = time.time()

    for q_grp in range(num_q_groups):
        for kv_h in range(num_kv_heads):
            # Preprocessor call
            func_preproc(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
            # QGEMM for each sub-block
            for sub in range(num_sub_blocks):
                func_qgemm(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)

    elapsed = time.time() - start

    print(f"  Total time:               {elapsed:.3f}s")
    print(f"  Per-qGEMM call:           {elapsed / total_qgemm * 1000:.4f} ms")
    print(f"  -> For 32 layers:         {elapsed * 32:.1f}s")

    return elapsed


if __name__ == "__main__":
    func_preproc, func_qgemm, preproc, codegen = compile_compass_operators()
    dev = tvm.cpu(0)

    print("\n" + "=" * 60)
    print("1. Single Call Latency")
    print("=" * 60)

    t_qgemm = bench_single_call(func_qgemm, codegen, dev)
    t_preproc = bench_single_preproc(func_preproc, preproc, dev)
    print(f"  Single qGEMM call:   {t_qgemm * 1000:.4f} ms")
    print(f"  Single preproc call: {t_preproc * 1000:.4f} ms")

    print("\n" + "=" * 60)
    print("2. COMPASS Loop Simulation (per layer, per chunk)")
    print("=" * 60)

    # Simulate different chunk scenarios from the 32K test:
    # Chunk 1: 1 block offloaded
    # Chunk 2: 2 blocks offloaded
    # ...
    # Chunk 7: 7 blocks offloaded
    for num_blocks in [1, 2, 3, 5, 7]:
        bench_compass_loop(func_preproc, func_qgemm, preproc, codegen, dev,
                           num_blocks=num_blocks)

    print("\n" + "=" * 60)
    print("3. Overhead Summary")
    print("=" * 60)

    # Calculate total for 32K (8 chunks, 32 layers)
    # Chunks 0-7 have 0,1,2,...,7 blocks respectively
    total_time = 0
    total_calls = 0
    for chunk_id in range(8):
        num_blocks = chunk_id  # chunk 0 has 0 blocks, chunk 1 has 1, etc.
        if num_blocks == 0:
            continue
        subs = num_blocks * 32
        calls = 32 * 8 * subs  # q_groups * kv_heads * sub_blocks
        preproc_calls = 32 * 8
        total_calls += calls + preproc_calls
        # Estimated time per call
        est_time = calls * t_qgemm + preproc_calls * t_preproc
        total_time += est_time
        print(f"  Chunk {chunk_id}: {num_blocks} blocks, {calls} qgemm calls, "
              f"est {est_time:.2f}s/layer, {est_time * 32:.1f}s for 32 layers")

    print(f"\n  Total est. TMAC time for 32K: {total_time * 32:.1f}s ({total_calls * 32} calls)")
    print(f"  BLASST total prefill time:    6.7s")
    print(f"  Ratio: {total_time * 32 / 6.7:.0f}x slower")
