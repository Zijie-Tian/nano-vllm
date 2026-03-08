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
logger.setLevel(logging.INFO)

def bench_qgemm(M, N, K, bits=2, num_threads=4, tune=False, n_trial=10):
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
    
    # Initialize Codegen - Using defaults now, should auto-detect x86
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
    
    # Compile
    print(f"\n[Bench] Compiling QGEMM (M={M}, N={N}, K={K}, bits={bits}, threads={num_threads})...")
    func, _ = codegen.compile(M, N, K, n_trial=n_trial)
    
    # Prepare data
    g = 4
    bm = codegen.bm
    _ngroups_per_elem = 8 // g
    
    A_shape = (M // bm, K // g, bm // _ngroups_per_elem)
    LUT_shape = (N, K // g, 1 << g)
    Scales_shape = (M // bm, K // codegen.group_size, bm // bits)
    LUT_Scales_shape = (N, K // codegen.act_group_size)
    LUT_Biases_shape = (N, K // codegen.act_group_size)
    C_shape = (N, M // bits)
    
    # use the actual out_dtype from codegen (should be float32 on x86)
    out_dtype = codegen.out_dtype
    print(f"  Using out_dtype: {out_dtype}")
    
    dev = tvm.cpu(0)
    A_tvm = tvm.nd.array(np.random.randint(0, 256, size=A_shape, dtype="uint8"), dev)
    LUT_tvm = tvm.nd.array(np.random.normal(size=LUT_shape).astype("int8"), dev)
    Scales_tvm = tvm.nd.array(np.random.normal(size=Scales_shape).astype(out_dtype), dev)
    LUT_Scales_tvm = tvm.nd.array(np.random.normal(size=LUT_Scales_shape).astype(out_dtype), dev)
    LUT_Biases_tvm = tvm.nd.array(np.random.normal(size=LUT_Biases_shape).astype(out_dtype), dev)
    C_tvm = tvm.nd.array(np.zeros(C_shape, dtype=out_dtype), dev)
    
    # Warmup
    for _ in range(10):
        func(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        func(A_tvm, LUT_tvm, Scales_tvm, LUT_Scales_tvm, LUT_Biases_tvm, C_tvm)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    tps = (M // bits * N) / avg_time
    print(f"  Avg Time: {avg_time*1000:.3f} ms")
    print(f"  Throughput: {tps/1e6:.3f} M-tokens/s")
    
    return avg_time

def bench_preprocessor(N, K, bits=2, num_threads=4, tune=False, n_trial=10):
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
    
    # Compile
    print(f"\n[Bench] Compiling Preprocessor (N={N}, K={K}, bits={bits}, threads={num_threads})...")
    func, _ = preprocessor.compile(N, K, n_trial=n_trial)
    
    # Prepare data
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
    
    # Warmup
    for _ in range(10):
        func(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        func(B_tvm, LUT_Scales_tvm, LUT_Biases_tvm, QLUT_tvm)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"  Avg Time: {avg_time*1000:.3f} ms")
    
    return avg_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run auto-tuning")
    args = parser.parse_args()

    # Test cases representing potential real-world scenarios
    # 2-bit, M=256, K=128, N=1
    bench_qgemm(M=256, N=1, K=128, bits=2, tune=args.tune)
    
    # 2-bit, M=256k*2, K=128, N=1
    bench_qgemm(M=256*1024*2, N=1, K=128, bits=2, tune=args.tune)
    
    bench_preprocessor(N=1, K=128, bits=2, tune=args.tune)
