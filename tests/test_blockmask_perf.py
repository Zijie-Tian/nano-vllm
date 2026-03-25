"""
test_blockmask_perf.py - FP32 blockmask performance benchmark on real kvcache-rope data

Usage:
    python tests/test_blockmask_perf.py <path_to_layer.pt>
    python tests/test_blockmask_perf.py   # auto-find layer_05.pt

Benchmarks qk_blockmask_fp32_omp with causal mask + scale=1/√D on each Q chunk.
Reports per-chunk and total timing, throughput, and skip%.
"""

import argparse
import glob
import math
import sys
import time

import torch

from nanovllm.sparse._cpu_ops import qk_blockmask_fp32_omp


def make_causal(BQ, BK, BS, STEP_KV, q_off):
    nq, nk = (BQ + BS - 1) // BS, (BK + STEP_KV - 1) // STEP_KV
    m = torch.zeros(nq, nk, dtype=torch.uint8)
    for qb in range(nq):
        qge = q_off + min((qb + 1) * BS, BQ)
        for kb in range(nk):
            if kb * STEP_KV < qge:
                m[qb, kb] = 1
    return m


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def run(data_path, lambda_val=0.1, chunk_size=4096, BS=128, STEP_KV=128,
        warmup=3, iters=10):
    data = torch.load(data_path, map_location="cpu")
    Q_all = data["post_rope_q"].float()
    K_all = data["post_rope_k"].float()
    S, n_heads, D = Q_all.shape
    n_kv_heads = K_all.shape[1]
    gqa = n_heads // n_kv_heads
    scale = 1.0 / D ** 0.5
    log_lambda = math.log(lambda_val)
    nc = (S + chunk_size - 1) // chunk_size

    print(f"Data: seq={S}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, D={D}")
    print(f"Config: λ={lambda_val}, scale=1/√{D}, chunk={chunk_size}, "
          f"BS={BS}, STEP_KV={STEP_KV}")
    print(f"Bench: warmup={warmup}, iters={iters}\n")

    header = (f"{'Chunk':>6} {'BQ':>6} {'BK':>8} {'Heads':>6} "
              f"{'Skip%':>7} {'Med_ms':>8} {'Blocks/s':>10}")
    print(header)
    print("-" * len(header))

    total_time = 0.0
    total_blocks = 0

    for c in range(nc):
        qs = c * chunk_size
        qe = min(qs + chunk_size, S)
        ke = qe
        BQ, BK = qe - qs, ke
        if BK < STEP_KV:
            continue

        causal = make_causal(BQ, BK, BS, STEP_KV, qs)
        eligible = causal.sum().item()

        # Run once for skip%
        total_keep = 0
        for kv_h in range(n_kv_heads):
            Q = Q_all[qs:qe, kv_h * gqa, :].contiguous()
            K = K_all[:ke, kv_h, :].contiguous()
            _, _, mask = qk_blockmask_fp32_omp(
                Q, K, log_lambda, scale=scale, input_mask=causal)
            total_keep += mask.sum().item()
        skip = (1.0 - total_keep / (eligible * n_kv_heads)) * 100

        # Benchmark: all heads together (realistic workload)
        def run_all_heads():
            for kv_h in range(n_kv_heads):
                Q = Q_all[qs:qe, kv_h * gqa, :].contiguous()
                K = K_all[:ke, kv_h, :].contiguous()
                qk_blockmask_fp32_omp(
                    Q, K, log_lambda, scale=scale, input_mask=causal)

        med_ms = bench(run_all_heads, warmup=warmup, iters=iters)
        blocks_per_sec = (eligible * n_kv_heads) / (med_ms / 1000)
        total_time += med_ms
        total_blocks += eligible * n_kv_heads

        print(f"{c:>6} {BQ:>6} {BK:>8} {n_kv_heads:>6} "
              f"{skip:>6.1f}% {med_ms:>7.1f} {blocks_per_sec:>9.0f}")

    print("-" * len(header))
    overall_bps = total_blocks / (total_time / 1000) if total_time > 0 else 0
    print(f"{'TOTAL':>6} {'':>6} {'':>8} {'':>6} "
          f"{'':>7} {total_time:>7.1f} {overall_bps:>9.0f}")

    print(f"\n=== Summary ===")
    print(f"  Total time (all chunks, {n_kv_heads} heads): {total_time:.1f}ms")
    print(f"  Total blocks processed: {total_blocks}")
    print(f"  Throughput: {overall_bps:.0f} blocks/s")


def main():
    parser = argparse.ArgumentParser(
        description="FP32 blockmask performance benchmark on kvcache-rope data"
    )
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to layer .pt file")
    parser.add_argument("--lambda", type=float, default=0.1, dest="lambda_val")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--step-kv", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        candidates = glob.glob(
            "results/kvcache-rope/**/layer_*.pt", recursive=True)
        if not candidates:
            print("ERROR: No layer .pt found.")
            sys.exit(1)
        data_path = candidates[0]

    print(f"Using: {data_path}\n")
    run(data_path, args.lambda_val, args.chunk_size, args.bs, args.step_kv,
        args.warmup, args.iters)


if __name__ == "__main__":
    main()
