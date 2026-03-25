# BLASST Block-Mask Test Suite

## Overview

Two test scripts for the `qk_blockmask_fp32_omp` CPU operator, using real post-RoPE KV cache data from Llama-3.1-8B. Both scripts use **causal block mask** + **scale=1/√D** to match real attention kernel behavior.

## Data Requirement

Requires `layer_XX.pt` files from the kvcache-rope dataset (stored at `results/kvcache-rope/`). Each file contains:
- `post_rope_q`: `[S, n_heads, D]` float16 — post-RoPE query cache
- `post_rope_k`: `[S, n_kv_heads, D]` float16 — post-RoPE key cache

## Scripts

### `test_blockmask_sparsity.py` — Lambda Sweep Sparsity Analysis

Sweeps multiple λ values and reports per-chunk skip% for each.

```bash
# Auto-detect data, default lambdas
python tests/test_blockmask_sparsity.py

# Specify data path
python tests/test_blockmask_sparsity.py results/kvcache-rope/.../layer_05.pt

# Custom lambdas
python tests/test_blockmask_sparsity.py --lambdas 0.1 0.001 0.0001 0.00001

# Custom chunk/block sizes
python tests/test_blockmask_sparsity.py --chunk-size 2048 --bs 64 --step-kv 64
```

**Sample output** (Llama-3.1-8B layer_05, 32k):
```
         λ      C0      C1      C2      C3      C4      C5      C6      C7  Overall
-------------------------------------------------------------------------------------
   0.10000   77.1%   88.0%   91.8%   93.5%   94.0%   94.8%   94.7%   95.6%    93.8%
   0.00100   26.1%   40.4%   52.4%   61.2%   66.3%   70.3%   73.1%   74.6%    66.7%
   0.00010    0.5%    5.5%   16.1%   27.6%   36.3%   44.0%   49.7%   53.7%    39.6%
   0.00001    0.0%    0.0%    0.7%    4.5%   10.6%   16.8%   22.9%   27.9%    15.9%
```

---

### `test_blockmask_perf.py` — FP32 Performance Benchmark

Benchmarks `qk_blockmask_fp32_omp` per-chunk with median timing and throughput.

```bash
# Default (λ=0.1, warmup=3, iters=10)
python tests/test_blockmask_perf.py

# Custom lambda and iterations
python tests/test_blockmask_perf.py --lambda 0.001 --warmup 5 --iters 20
```

**Sample output** (24-core OMP, Llama-3.1-8B layer_05):
```
 Chunk     BQ       BK  Heads   Skip%   Med_ms   Blocks/s
---------------------------------------------------------
     0   4096     4096      8   77.1%   150.5     28073
     3   4096    16384      8   93.5%   916.8     31415
     7   3813    32485      8   95.6%  1882.5     30534
---------------------------------------------------------
 TOTAL                                 8276.4     31304
```

## Environment

```bash
source build/nano-vllm-envs.sh
OMP_NUM_THREADS=24 taskset -c 0,2,4,... python tests/test_blockmask_XXX.py
```

> [!IMPORTANT]
> Pin OMP threads to physical cores for consistent benchmarking. Use `taskset` to avoid hyperthreading interference.
