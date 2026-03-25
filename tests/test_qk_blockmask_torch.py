"""
test_qk_blockmask_torch.py - Correctness + performance test for torch blockmask ops

Uses real kvcache-rope data (layer_05.pt) with causal mask + scale=1/√D.
Tests FP32 OMP and VNNI OMP against pure-torch reference.
"""

import math
import sys
import time
import glob

import torch

from nanovllm.sparse._cpu_ops import (
    pack_k_vnni,
    qk_blockmask_fp32_omp,
    qk_blockmask_vnni_omp,
)


def make_causal_block_mask(BQ, BK, BS=128, STEP_KV=128, q_offset=0):
    """Block-level causal mask [n_q_blocks, n_k_blocks]."""
    nq = (BQ + BS - 1) // BS
    nk = (BK + STEP_KV - 1) // STEP_KV
    mask = torch.zeros(nq, nk, dtype=torch.uint8)
    for qb in range(nq):
        q_global_end = q_offset + min((qb + 1) * BS, BQ)
        for kb in range(nk):
            if kb * STEP_KV < q_global_end:
                mask[qb, kb] = 1
    return mask


def torch_reference_blockmask(Q, K, log_lambda, scale=1.0,
                              input_mask=None, BS=128, STEP_KV=128):
    """Pure-torch reference with scale + input_mask."""
    BQ, D = Q.shape
    BK = K.shape[0]
    nk = (BK + STEP_KV - 1) // STEP_KV
    nq = (BQ + BS - 1) // BS

    block_rowmax = torch.empty(BQ, nk)
    running_max = torch.full((BQ,), float("-inf"))
    block_mask = torch.empty(nq, nk, dtype=torch.uint8)

    for kblk in range(nk):
        k_s = kblk * STEP_KV
        k_e = min(k_s + STEP_KV, BK)
        for qblk in range(nq):
            q_s = qblk * BS
            q_e = min(q_s + BS, BQ)
            if input_mask is not None and input_mask[qblk, kblk] == 0:
                block_rowmax[q_s:q_e, kblk] = float("-inf")
                block_mask[qblk, kblk] = 0
                continue
            S = Q[q_s:q_e] @ K[k_s:k_e].T * scale
            lm = S.max(dim=1).values
            block_rowmax[q_s:q_e, kblk] = lm
            running_max[q_s:q_e] = torch.max(running_max[q_s:q_e], lm)
            diff = block_rowmax[q_s:q_e, kblk] - running_max[q_s:q_e]
            block_mask[qblk, kblk] = 0 if (diff < log_lambda).all().item() else 1
    return block_rowmax, running_max, block_mask


def load_real_data():
    """Load layer_05.pt from kvcache-rope dataset."""
    candidates = glob.glob("results/kvcache-rope/**/layer_05.pt", recursive=True)
    if not candidates:
        print("ERROR: layer_05.pt not found. Download from aliyunpan first.")
        sys.exit(1)
    data = torch.load(candidates[0], map_location="cpu")
    Q_all = data["post_rope_q"].float()  # [S, n_heads, D]
    K_all = data["post_rope_k"].float()  # [S, n_kv_heads, D]
    print(f"Loaded: {candidates[0]}")
    S, n_heads, D = Q_all.shape
    _, n_kv_heads, _ = K_all.shape
    print(f"  seq_len={S}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, D={D}")
    return Q_all, K_all


def extract_chunk(Q_all, K_all, chunk_idx, chunk_size=4096, kv_head=0):
    """Extract a (Q, K) pair for a given chunk and KV head."""
    S = Q_all.shape[0]
    n_kv_heads = K_all.shape[1]
    gqa_ratio = Q_all.shape[1] // n_kv_heads
    q_start = chunk_idx * chunk_size
    q_end = min(q_start + chunk_size, S)
    k_end = q_end
    Q = Q_all[q_start:q_end, kv_head * gqa_ratio, :].contiguous()
    K = K_all[:k_end, kv_head, :].contiguous()
    return Q, K, q_start


def test_correctness(Q_all, K_all):
    """Verify FP32 OMP and VNNI OMP against torch reference on real data."""
    print("=" * 60)
    print("Correctness (real data, causal + scale=1/√D)")
    print("=" * 60)

    D = Q_all.shape[2]
    scale = 1.0 / D ** 0.5
    BS, STEP_KV = 128, 128
    log_lambda = math.log(0.1)

    # Test on chunk 3 (mid-sequence, ~16k context) with kv_head 0
    Q, K, q_offset = extract_chunk(Q_all, K_all, chunk_idx=3, kv_head=0)
    BQ, BK = Q.shape[0], K.shape[0]
    causal = make_causal_block_mask(BQ, BK, BS, STEP_KV, q_offset=q_offset)
    eligible = causal.sum().item()

    print(f"  Chunk 3: Q=[{q_offset}:{q_offset+BQ}], K=[0:{BK}], "
          f"eligible={eligible}/{causal.numel()}")

    # --- Torch reference ---
    ref_brm, ref_rmax, ref_mask = torch_reference_blockmask(
        Q, K, log_lambda, scale=scale, input_mask=causal)

    # --- FP32 OMP ---
    fp32_brm, fp32_rmax, fp32_mask = qk_blockmask_fp32_omp(
        Q, K, log_lambda, scale=scale, input_mask=causal)

    diff = fp32_brm - ref_brm
    finite = torch.isfinite(diff)
    brm_err = diff[finite].abs().max().item() if finite.any() else 0.0
    mask_diffs = (fp32_mask != ref_mask).sum().item()
    fp32_skip = (1 - fp32_mask.sum().item() / eligible) * 100

    print(f"  [FP32 OMP] brm_err={brm_err:.6f}  mask_diffs={mask_diffs}  "
          f"skip={fp32_skip:.1f}%")
    assert brm_err < 1e-4, f"BRM error too large: {brm_err}"
    assert mask_diffs == 0, f"Mask mismatch: {mask_diffs}"
    for qb in range(causal.shape[0]):
        for kb in range(causal.shape[1]):
            if causal[qb, kb] == 0:
                assert fp32_mask[qb, kb] == 0
    print("  [PASSED] FP32 OMP\n")

    # --- VNNI OMP (only if D is small enough for VNNI microkernel) ---
    if D <= 32:
        K_vnni, scale_k, sum_k = pack_k_vnni(K, 4096)
        vnni_brm, vnni_rmax, vnni_mask = qk_blockmask_vnni_omp(
            Q, K_vnni, scale_k, sum_k, log_lambda,
            scale=scale, input_mask=causal)
        vdiff = vnni_brm - ref_brm
        vfinite = torch.isfinite(vdiff)
        vnni_rel = (vdiff[vfinite].abs() / (ref_brm[vfinite].abs() + 1e-6)
                    ).max().item() if vfinite.any() else 0.0
        vmask_diffs = (vnni_mask != ref_mask).sum().item()
        print(f"  [VNNI OMP] rel_err={vnni_rel*100:.2f}%  mask_diffs={vmask_diffs}")
        assert vnni_rel < 0.1
        print("  [PASSED] VNNI OMP\n")
    else:
        print(f"  [SKIP] VNNI OMP (D={D} > 32, exceeds microkernel design)\n")


def test_multi_chunk(Q_all, K_all):
    """Test across multiple chunks and KV heads for comprehensive coverage."""
    print("=" * 60)
    print("Multi-chunk verification (3 chunks × 3 KV heads)")
    print("=" * 60)

    D = Q_all.shape[2]
    n_kv_heads = K_all.shape[1]
    scale = 1.0 / D ** 0.5
    BS, STEP_KV = 128, 128
    log_lambda = math.log(0.1)
    S = Q_all.shape[0]
    n_chunks = (S + 4096 - 1) // 4096

    total_diffs = 0
    total_blocks = 0

    print(f"{'Chunk':>6} {'KV_h':>6} {'BRM_err':>10} {'Diffs':>6} {'Skip%':>7}")
    print("-" * 40)

    for c in [0, 3, min(7, n_chunks - 1)]:
        for kv_h in [0, n_kv_heads // 2, n_kv_heads - 1]:
            Q, K, q_off = extract_chunk(Q_all, K_all, c, kv_head=kv_h)
            BQ, BK = Q.shape[0], K.shape[0]
            if BK < STEP_KV:
                continue
            causal = make_causal_block_mask(BQ, BK, BS, STEP_KV, q_offset=q_off)
            eligible = causal.sum().item()

            cpp_brm, _, cpp_mask = qk_blockmask_fp32_omp(
                Q, K, log_lambda, scale=scale, input_mask=causal)
            ref_brm, _, ref_mask = torch_reference_blockmask(
                Q, K, log_lambda, scale=scale, input_mask=causal)

            d = cpp_brm - ref_brm
            f = torch.isfinite(d)
            err = d[f].abs().max().item() if f.any() else 0.0
            diffs = (cpp_mask != ref_mask).sum().item()
            skip = (1 - cpp_mask.sum().item() / eligible) * 100

            total_diffs += diffs
            total_blocks += ref_mask.numel()

            print(f"{c:>6} {kv_h:>6} {err:>10.6f} {diffs:>6} {skip:>6.1f}%")

    print("-" * 40)
    print(f"Total mask diffs: {total_diffs}/{total_blocks}")
    assert total_diffs == 0, f"Found {total_diffs} mask mismatches!"
    print("[PASSED]\n")


def benchmark_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def test_performance(Q_all, K_all):
    """Benchmark on real data chunks."""
    print("=" * 60)
    print("Performance (real data, causal + scale=1/√D)")
    print("=" * 60)

    D = Q_all.shape[2]
    scale = 1.0 / D ** 0.5
    BS, STEP_KV = 128, 128
    log_lambda = math.log(0.1)

    print(f"\n{'Chunk':>6} {'BQ':>6} {'BK':>8} {'FP32_ms':>8} {'Skip%':>7}")
    print("-" * 40)

    for c in [0, 3, 7]:
        Q, K, q_off = extract_chunk(Q_all, K_all, c, kv_head=0)
        BQ, BK = Q.shape[0], K.shape[0]
        if BK < STEP_KV:
            continue
        causal = make_causal_block_mask(BQ, BK, BS, STEP_KV, q_offset=q_off)
        eligible = causal.sum().item()

        _, _, mask = qk_blockmask_fp32_omp(
            Q, K, log_lambda, scale=scale, input_mask=causal)
        skip = (1 - mask.sum().item() / eligible) * 100

        ms = benchmark_fn(
            lambda: qk_blockmask_fp32_omp(
                Q, K, log_lambda, scale=scale, input_mask=causal),
            warmup=2, iters=5
        )
        print(f"{c:>6} {BQ:>6} {BK:>8} {ms:>7.1f} {skip:>6.1f}%")

    print()


if __name__ == "__main__":
    Q_all, K_all = load_real_data()
    print()
    test_correctness(Q_all, K_all)
    test_multi_chunk(Q_all, K_all)
    test_performance(Q_all, K_all)
    print("All tests passed!")
