"""
test_blockmask_sparsity.py - BLASST blockmask sparsity analysis on real kvcache-rope data

Usage:
    python tests/test_blockmask_sparsity.py <path_to_layer.pt>
    python tests/test_blockmask_sparsity.py   # auto-find layer_05.pt

Uses causal block mask + scale=1/√D. Reports per-chunk skip% for each λ.
"""

import argparse
import glob
import math
import sys

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


def run(data_path, lambdas, chunk_size=4096, BS=128, STEP_KV=128):
    data = torch.load(data_path, map_location="cpu")
    Q_all = data["post_rope_q"].float()
    K_all = data["post_rope_k"].float()
    S, n_heads, D = Q_all.shape
    n_kv_heads = K_all.shape[1]
    gqa = n_heads // n_kv_heads
    scale = 1.0 / D ** 0.5
    nc = (S + chunk_size - 1) // chunk_size

    print(f"Data: seq={S}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
          f"D={D}, scale=1/√{D}")
    print(f"Causal mask + scale, chunk_size={chunk_size}, "
          f"BS={BS}, STEP_KV={STEP_KV}\n")

    # Header: per-chunk columns + overall
    chunk_ids = list(range(nc))
    chunk_labels = [f"C{c}" for c in chunk_ids]
    print(f"{'λ':>10} " + " ".join(f"{l:>7}" for l in chunk_labels)
          + f" {'Overall':>8}")
    print("-" * (12 + 8 * len(chunk_ids) + 9))

    for lam in lambdas:
        ll = math.log(lam)
        chunk_skips = {}
        total_keep, total_elig = 0, 0

        for c in range(nc):
            qs = c * chunk_size
            qe = min(qs + chunk_size, S)
            ke = qe
            BQ, BK = qe - qs, ke
            if BK < STEP_KV:
                continue
            causal = make_causal(BQ, BK, BS, STEP_KV, qs)
            ck, ce = 0, 0
            for kv_h in range(n_kv_heads):
                Q = Q_all[qs:qe, kv_h * gqa, :].contiguous()
                K = K_all[:ke, kv_h, :].contiguous()
                _, _, mask = qk_blockmask_fp32_omp(
                    Q, K, ll, scale=scale, input_mask=causal
                )
                ck += mask.sum().item()
                ce += causal.sum().item()
            total_keep += ck
            total_elig += ce
            chunk_skips[c] = (1.0 - ck / ce) * 100 if ce > 0 else 0

        overall = (1.0 - total_keep / total_elig) * 100 if total_elig > 0 else 0
        cols = [f"{chunk_skips.get(c, 0):>6.1f}%" for c in chunk_ids]
        print(f"{lam:>10.5f} " + " ".join(cols) + f" {overall:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="BLASST blockmask sparsity analysis on kvcache-rope data"
    )
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to layer .pt file (auto-detect if omitted)")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.1, 0.001, 0.0001, 0.00001],
                        help="Lambda values to sweep")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--step-kv", type=int, default=128)
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        candidates = glob.glob(
            "results/kvcache-rope/**/layer_*.pt", recursive=True
        )
        if not candidates:
            print("ERROR: No layer .pt found. Pass path explicitly.")
            sys.exit(1)
        data_path = candidates[0]

    print(f"Using: {data_path}\n")
    run(data_path, args.lambdas, args.chunk_size, args.bs, args.step_kv)


if __name__ == "__main__":
    main()
