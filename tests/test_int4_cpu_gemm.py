"""
Benchmark: _weight_int4pack_mm_for_cpu vs FP32 matmul on CPU.
Measures single-GEMM throughput and extrapolates to COMPASS scenarios.

COMPASS hot path per layer:
  G=32 groups, each doing:
    for h in range(H=4):  # kv_heads
        Q_h: [hpg*F, D] = [896, 128]
        K_h: [N_kv, D]  (N_kv varies by context length)
        scores = Q_h @ K_h.T  → [896, N_kv]

  Total FLOPs per layer = G * H * 2 * (hpg*F) * D * N_kv

Usage:
    python tests/test_int4_cpu_gemm.py
"""
import time
import torch


def quantize_to_int4_packed(weight_fp, groupsize=128):
    """Quantize [N, K] float tensor to int4 packed format for CPU tinygemm."""
    N, K = weight_fp.shape
    assert K % groupsize == 0
    num_groups = K // groupsize
    w_grouped = weight_fp.reshape(N, num_groups, groupsize)
    w_min = w_grouped.amin(dim=-1)
    w_max = w_grouped.amax(dim=-1)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-6)
    mid_point = 8.0
    zero_point = w_min + scale * mid_point
    min_val = w_min.unsqueeze(-1)
    scale_exp = scale.unsqueeze(-1)
    int_data = torch.clamp(
        torch.round((w_grouped - min_val) / scale_exp), 0, 15
    ).to(torch.int32).reshape(N, K)
    packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(int_data, 1)
    sz = torch.cat([scale.to(torch.bfloat16).unsqueeze(-1),
                    zero_point.to(torch.bfloat16).unsqueeze(-1)], dim=-1)
    scale_and_zero = sz.transpose(0, 1).contiguous()
    return packed_weight, scale_and_zero


def bench(fn, warmup=10, repeat=50):
    """Return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 70)
    print("Int4 vs FP32 CPU GEMM: Single-call throughput + COMPASS extrapolation")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}  |  CPU threads: {torch.get_num_threads()}")

    # ================================================================
    # Part 1: Single GEMM throughput at different N (KV token count)
    # ================================================================
    # Fixed: M = hpg * F = 7 * 128 = 896, K = D = 128
    M = 896
    K = 128
    N_values = [1024, 2048, 4096, 8192, 16384, 32768]

    print(f"\n--- Part 1: Single GEMM [M={M}, K={K}] @ [N, K]^T ---")
    print(f"{'N (tokens)':<12} {'FP32 (ms)':>10} {'Int4 (ms)':>10} {'Pack (ms)':>10} {'Int4/FP32':>10} {'Int4 GFLOPS':>12}")
    print("-" * 66)

    throughputs_fp32 = {}
    throughputs_int4 = {}

    for N in N_values:
        torch.manual_seed(42)
        q_fp32 = torch.randn(M, K, dtype=torch.float32)
        k_fp32 = torch.randn(N, K, dtype=torch.float32)
        q_bf16 = q_fp32.bfloat16()

        # FP32 matmul
        fp32_ms = bench(lambda: torch.mm(q_fp32, k_fp32.T))

        # Int4: quantize + pack (one-time)
        pack_ms = bench(lambda: quantize_to_int4_packed(k_fp32, groupsize=K), warmup=3, repeat=10)
        pw, sz = quantize_to_int4_packed(k_fp32, groupsize=K)

        # Int4 matmul
        int4_ms = bench(lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(
            q_bf16.contiguous(), pw, K, sz))

        ratio = fp32_ms / int4_ms
        flops = 2.0 * M * N * K
        gflops_int4 = flops / (int4_ms * 1e-3) / 1e9

        throughputs_fp32[N] = fp32_ms
        throughputs_int4[N] = int4_ms

        print(f"{N:<12} {fp32_ms:>10.3f} {int4_ms:>10.3f} {pack_ms:>10.3f} {ratio:>9.2f}x {gflops_int4:>11.1f}")

    # ================================================================
    # Part 2: Extrapolate to COMPASS full-layer cost
    # ================================================================
    # Qwen2-7B: H=4 kv_heads, hpg=7, D=128, G=32 q-groups, 28 layers
    H = 4
    G = 32
    num_layers = 28

    print(f"\n--- Part 2: COMPASS extrapolation (Qwen2-7B: H={H}, G={G}) ---")
    print(f"  Per-layer cost = G × H × single_gemm_time = {G} × {H} × t")
    print(f"  Full-model cost = per_layer × {num_layers}")
    print()

    block_configs = [
        (2,   "8K"),
        (4,   "16K"),
        (8,   "32K"),
        (16,  "64K"),
        (32,  "128K"),
    ]

    print(f"{'Blocks':<8} {'KV tokens':<12} {'N_kv':<10} "
          f"{'FP32/layer':>12} {'Int4/layer':>12} {'Speedup':>8} "
          f"{'FP32/model':>12} {'Int4/model':>12}")
    print("-" * 96)

    for num_blocks, label in block_configs:
        N_kv = num_blocks * 4096  # total KV tokens
        # Find closest measured N or interpolate
        if N_kv in throughputs_fp32:
            t_fp32 = throughputs_fp32[N_kv]
            t_int4 = throughputs_int4[N_kv]
        else:
            # Linear interpolation from measured points
            ns = sorted(throughputs_fp32.keys())
            if N_kv <= ns[0]:
                t_fp32 = throughputs_fp32[ns[0]] * N_kv / ns[0]
                t_int4 = throughputs_int4[ns[0]] * N_kv / ns[0]
            elif N_kv >= ns[-1]:
                t_fp32 = throughputs_fp32[ns[-1]] * N_kv / ns[-1]
                t_int4 = throughputs_int4[ns[-1]] * N_kv / ns[-1]
            else:
                for i in range(len(ns) - 1):
                    if ns[i] <= N_kv <= ns[i + 1]:
                        frac = (N_kv - ns[i]) / (ns[i + 1] - ns[i])
                        t_fp32 = throughputs_fp32[ns[i]] * (1 - frac) + throughputs_fp32[ns[i + 1]] * frac
                        t_int4 = throughputs_int4[ns[i]] * (1 - frac) + throughputs_int4[ns[i + 1]] * frac
                        break

        layer_fp32 = G * H * t_fp32
        layer_int4 = G * H * t_int4
        model_fp32 = layer_fp32 * num_layers
        model_int4 = layer_int4 * num_layers
        speedup = layer_fp32 / layer_int4

        print(f"{num_blocks:<8} {label:<12} {N_kv:<10} "
              f"{layer_fp32:>10.0f}ms {layer_int4:>10.0f}ms {speedup:>7.2f}x "
              f"{model_fp32/1000:>10.1f}s {model_int4/1000:>10.1f}s")

    # ================================================================
    # Part 3: Correctness check
    # ================================================================
    print(f"\n--- Part 3: Correctness ---")
    torch.manual_seed(42)
    q = torch.randn(M, K, dtype=torch.float32)
    k = torch.randn(4096, K, dtype=torch.float32)
    ref = torch.mm(q, k.T)
    pw, sz = quantize_to_int4_packed(k, groupsize=K)
    out = torch.ops.aten._weight_int4pack_mm_for_cpu(q.bfloat16().contiguous(), pw, K, sz)
    cos = torch.nn.functional.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0)
    print(f"  Cosine similarity (FP32 vs Int4): {cos.item():.6f}")
    assert cos > 0.99, f"FAILED: {cos.item()}"
    print("  correctness: PASSED")


if __name__ == "__main__":
    main()
