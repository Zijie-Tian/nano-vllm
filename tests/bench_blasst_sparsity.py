import torch
import os
import triton
import math
import time
from nanovllm.ops.blasst_chunked_prefill import blasst_chunked_prefill


def load_real_data(ctx_len="32k"):
    """Load real QKV traces from COMPASS results."""
    path = f"tests/data/real_kvcache/{ctx_len}/layer_05.pt"
    if not os.path.exists(path):
        print(
            f"Data not found at {path}. Please symlink /home/zijie/Code/COMPASS/results/kvcache-rope/glm-4-9b to tests/data/real_kvcache"
        )
        return None, None, None
    data = torch.load(path)
    # Shape: [Tokens, Heads, Dim] -> [1, Heads, Tokens, Dim]
    q = data["post_rope_q"].unsqueeze(0).transpose(1, 2).cuda().half()
    k = data["post_rope_k"].unsqueeze(0).transpose(1, 2).cuda().half()
    v = data["v"].unsqueeze(0).transpose(1, 2).cuda().half()
    return q, k, v


def run_bench_on_data(q, k, v, lambda_val, iters=5):
    """Run BLASST benchmark simulating a multi-block prefill pipeline."""
    # Simulation parameters
    q_start, q_end = 8192, 16384
    q_chunk = q[:, :, q_start:q_end, :].contiguous()
    batch, heads, q_len, head_dim = q_chunk.shape

    BLOCK_M, BLOCK_N = 128, 64
    grid_0 = triton.cdiv(q_len, BLOCK_M)
    grid_1 = batch * heads

    threshold_ln = math.log(lambda_val)

    # Simulate processing 3 blocks: 2 historical, 1 causal
    kv_ranges = [(0, 4096, False), (4096, 8192, False), (8192, 16384, True)]

    total_time = 0

    all_masks = []

    # We run multiple iterations for timing, but statistics are taken from the last one
    for _ in range(iters):
        historical_lse = None
        current_iter_time = 0
        current_iter_masks = []

        for start, end, is_causal in kv_ranges:
            k_block = k[:, :, start:end, :].contiguous()
            v_block = v[:, :, start:end, :].contiguous()
            num_kv_sub = triton.cdiv(end - start, BLOCK_N)

            # 1. Initialize In-Out Mask
            mask = torch.ones(
                (grid_0, grid_1, num_kv_sub), device="cuda", dtype=torch.int8
            )
            if is_causal:
                # Apply block-level causal mask
                for q_idx in range(grid_0):
                    q_end_pos = (q_idx + 1) * BLOCK_M
                    for kv_idx in range(num_kv_sub):
                        if kv_idx * BLOCK_N >= q_end_pos:
                            mask[q_idx, :, kv_idx] = 0

            # 2. Compute
            torch.cuda.synchronize()
            t0 = time.time()
            _, historical_lse = blasst_chunked_prefill(
                q_chunk,
                k_block,
                v_block,
                threshold_ln_lambda=threshold_ln,
                lse_in=historical_lse,
                mask_buffer=mask,
            )
            torch.cuda.synchronize()
            current_iter_time += time.time() - t0
            current_iter_masks.append(mask)

        total_time += current_iter_time
        all_masks = current_iter_masks

    # Calculate statistics from the last iteration
    full_mask = torch.cat(all_masks, dim=2)
    compute_density = full_mask.float().mean().item()
    required_kv_density = full_mask.any(dim=0).float().mean().item()
    avg_time_ms = (total_time / iters) * 1000

    return {
        "lambda": lambda_val,
        "compute_density": compute_density * 100,
        "required_kv_density": required_kv_density * 100,
        "time_ms": avg_time_ms,
    }


def main():
    print("=== BLASST Sparsity & Performance Benchmark (Real Data) ===")
    print("Testing with LSE Propagation and In-Out Causal Masking")

    q, k, v = load_real_data("32k")
    if q is None:
        return

    lambdas = [1.2, 0.8, 0.5, 0.1, 0.01, 0.001]

    print(
        f"\n{'Lambda':>8} | {'Compute Density':>16} | {'Required KV (IO)':>16} | {'Time (ms)':>10}"
    )
    print("-" * 65)

    for lambda_val in lambdas:
        res = run_bench_on_data(q, k, v, lambda_val)
        print(
            f"{res['lambda']:8.3f} | {res['compute_density']:15.2f}% | {res['required_kv_density']:15.2f}% | {res['time_ms']:9.2f}"
        )


if __name__ == "__main__":
    main()
