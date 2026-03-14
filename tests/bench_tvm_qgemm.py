import numpy as np
import tvm
import torch
import time
import sys
import os
import logging

# Ensure nanovllm is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm.ops.tvm_qgemm.qgemm import (
    QGeMMLUTBitsCodegen,
    QGeMMLUTBitsPreprocessorCodegen,
)

# Setup logging
logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger("bench_qgemm")
logger.setLevel(logging.WARNING)


def bench_qgemm(M, N, K, bits=2, num_threads=4, tune=False, n_trial=1):
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"

    codegen = QGeMMLUTBitsCodegen(
        dtype="int8",
        target=target,
        name="qgemm_bench",
        bits=bits,
        num_threads=num_threads,
        save_dir="build/bench_qgemm",
        verify=False,
        tune=tune,
        reuse_tuned=True,
    )

    print(f"\n[Bench] QGEMM (M={M}, N={N}, K={K}, bits={bits}, threads={num_threads})")
    func, _ = codegen.compile(M, N, K, n_trial=n_trial)

    g = 4
    bm = codegen.bm
    _ngroups_per_elem = 8 // g

    A_shape = (M // bm, K // g, bm // _ngroups_per_elem)
    LUT_shape = (N, K // g, 1 << g)
    Scales_shape = (M // bm, K // codegen.group_size, bm // bits)
    LUT_Scales_shape = (N, K // codegen.act_group_size)
    LUT_Biases_shape = (N, K // codegen.act_group_size)
    C_shape = (N, M // bits)

    out_dtype = codegen.out_dtype
    dev = tvm.cpu(0)
    A_tvm = tvm.nd.array(np.random.randint(0, 256, size=A_shape, dtype="uint8"), dev)
    LUT_tvm = tvm.nd.array(np.random.normal(size=LUT_shape).astype("int8"), dev)
    Scales_tvm = tvm.nd.array(
        np.random.normal(size=Scales_shape).astype(out_dtype), dev
    )
    LUT_Scales_tvm = tvm.nd.array(
        np.random.normal(size=LUT_Scales_shape).astype(out_dtype), dev
    )
    LUT_Biases_tvm = tvm.nd.array(
        np.random.normal(size=LUT_Biases_shape).astype(out_dtype), dev
    )
    C_tvm = tvm.nd.array(np.zeros(C_shape, dtype=out_dtype), dev)

    for _ in range(5):
        func(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)

    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        func(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs * 1000
    print(f"  Latency: {avg_latency:.4f} ms")

    return avg_latency


def bench_preprocessor(N, K, bits=2, num_threads=4, tune=False, n_trial=1):
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"

    preprocessor = QGeMMLUTBitsPreprocessorCodegen(
        dtype="int8",
        target=target,
        name="preproc_bench",
        bits=bits,
        num_threads=num_threads,
        save_dir="build/bench_qgemm",
        verify=False,
        tune=tune,
        reuse_tuned=True,
    )

    print(f"\n[Bench] Preprocessor (N={N}, K={K}, bits={bits}, threads={num_threads})")
    func, _ = preprocessor.compile(N, K, n_trial=n_trial)

    B_shape = (N, K)
    dev = tvm.cpu(0)
    out_dtype = preprocessor.out_dtype
    B_tvm = tvm.nd.array(np.random.normal(size=B_shape).astype(out_dtype), dev)

    LUT_Scales_shape = (N, K // preprocessor.act_group_size)
    LUT_Biases_shape = (N, K // preprocessor.act_group_size)
    QLUT_shape = (N, K // preprocessor.g, 1 << preprocessor.g)

    LUT_Scales_tvm = tvm.nd.array(np.zeros(LUT_Scales_shape, dtype=out_dtype), dev)
    LUT_Biases_tvm = tvm.nd.array(np.zeros(LUT_Biases_shape, dtype=out_dtype), dev)
    QLUT_tvm = tvm.nd.array(np.zeros(QLUT_shape, dtype="int8"), dev)

    for _ in range(5):
        func(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)

    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        func(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs * 1000
    print(f"  Latency: {avg_latency:.4f} ms")

    return avg_latency


def bench_torch_fp32_gemm(seq_len, N, K, num_threads=None):
    """Benchmark equivalent FP32 CPU torch matmul: Q[N, K] @ K_data[seq_len, K]^T.

    This is the dense FP32 baseline that TMAC 2-bit qGEMM replaces.
    seq_len corresponds to M // bits in the qGEMM benchmark.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    actual_threads = torch.get_num_threads()

    print(f"\n[Bench] Torch FP32 CPU GEMM (seq_len={seq_len}, N={N}, K={K}, threads={actual_threads})")

    Q = torch.randn(N, K, dtype=torch.float32)
    K_data = torch.randn(seq_len, K, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = torch.mm(Q, K_data.t())

    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        _ = torch.mm(Q, K_data.t())
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs * 1000
    print(f"  Latency: {avg_latency:.4f} ms")

    return avg_latency


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run auto-tuning")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads for torch (default: all cores)")
    args = parser.parse_args()

    BITS = 2
    # COMPASS-realistic: K=128 (head_dim), N=1 (per-head)
    # Ultra-fast: K=1024 (hidden_dim), N=4096 (query chunk)
    configs = [
        # (label, N, K, M_values)
        ("COMPASS Per-Head (N=1, K=128)", 1, 128,
         [256, 1024, 4096, 8192, 16384, 32768]),
        ("Ultra-Fast QK (N=4096, K=1024)", 4096, 1024,
         [32768, 65536, 131072, 262144, 524288, 1048576]),
    ]

    for label, N_q, K_dim, M_values in configs:
        print(f"\n{'=' * 78}")
        print(f"  {label}, bits={BITS}")
        print(f"{'=' * 78}")

        header = (f"{'M (TVM)':>10} | {'seq_len':>10} | {'TVM qGEMM':>12} | "
                  f"{'Torch FP32':>12} | {'Speedup':>8}")
        print(f"\n{header}")
        print("-" * len(header))

        for M_val in M_values:
            seq_len = M_val // BITS

            # TVM qGEMM benchmark
            t_tvm = bench_qgemm(
                M=M_val, N=N_q, K=K_dim, bits=BITS, tune=args.tune)

            # FP32 CPU torch baseline
            t_torch = bench_torch_fp32_gemm(
                seq_len=seq_len, N=N_q, K=K_dim, num_threads=args.threads)

            speedup = t_torch / t_tvm if t_tvm > 0 else float('inf')

            print(f"\n  {M_val:>10} | {seq_len:>10} | {t_tvm:>9.4f} ms | "
                  f"{t_torch:>9.4f} ms | {speedup:>6.2f}x")

    # Preprocessor bench
    print(f"\n{'=' * 78}")
    print(f"  Preprocessor Bench")
    print(f"{'=' * 78}")
    for label, N_q, K_dim, _ in configs:
        bench_preprocessor(N=N_q, K=K_dim, bits=BITS, tune=args.tune)
