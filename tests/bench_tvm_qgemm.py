import numpy as np
import tvm
from tvm import autotvm
import time
import sys
import os
import logging

# Ensure nanovllm is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanovllm.ops.tvm_qgemm.qgemm import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from nanovllm.ops.tvm_qgemm.utils.math_utils import nmse

# Setup logging
logging.basicConfig(format='%(levelname)s: %(message)s')
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
        reuse_tuned=True
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
    Scales_tvm = tvm.nd.array(np.random.normal(size=Scales_shape).astype(out_dtype), dev)
    LUT_Scales_tvm = tvm.nd.array(np.random.normal(size=LUT_Scales_shape).astype(out_dtype), dev)
    LUT_Biases_tvm = tvm.nd.array(np.random.normal(size=LUT_Biases_shape).astype(out_dtype), dev)
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
        reuse_tuned=True
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run auto-tuning")
    args = parser.parse_args()

    # N=4096 (query chunk size), K=1024 (hidden_dim), M=seq_len (32k to 1M)
    K_dim = 1024
    N_queries = 4096
    seq_lengths = [32768, 65536, 131072, 262144, 524288, 1048576]
    
    print(f"=== Ultra-Fast QGEMM QK Bench (N={N_queries}, K={K_dim}, 2-bit) ===")
    for m in seq_lengths:
        bench_qgemm(M=m, N=N_queries, K=K_dim, bits=2, tune=args.tune)
    
    print("\n=== Ultra-Fast Preprocessor Bench (Query to LUT) ===")
    bench_preprocessor(N=N_queries, K=K_dim, bits=2, tune=args.tune)
