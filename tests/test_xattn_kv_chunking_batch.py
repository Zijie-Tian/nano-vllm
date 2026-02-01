"""
Test: 批量验证 xattn_estimate 与 KV chunking kernels 的一致性

测试 results/kvcache 下所有保存的 QKV 数据

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
        python tests/test_xattn_kv_chunking_batch.py
"""
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

import os
import glob
import torch
import math
from nanovllm.ops.xattn import (
    xattn_estimate,
    flat_group_gemm_fuse_reshape,
    softmax_compute_partial_stats,
    softmax_normalize_and_block_sum,
    merge_softmax_stats,
    find_blocks_chunked,
)

# ============================================================
# 参数配置
# ============================================================
DATA_DIR = "/home/zijie/Code/nano-vllm/results/kvcache"
BSA_BLOCK_SIZE = 128
CHUNK_SIZE = 16384

device = "cuda"


def test_single_file(data_file: str) -> dict:
    """测试单个 kvcache 文件"""
    data = torch.load(data_file, map_location="cpu")
    Q = data["query"].to(device)
    K = data["key"].to(device)

    batch_size, num_heads, seq_len, head_dim = Q.shape
    STRIDE = data["stride"]
    THRESHOLD = data["threshold"][0].item() if isinstance(data["threshold"], torch.Tensor) else data["threshold"]

    # ========== xattn_estimate API ==========
    attn_sums_api, mask_api = xattn_estimate(
        Q, K,
        block_size=BSA_BLOCK_SIZE,
        stride=STRIDE,
        threshold=THRESHOLD,
        chunk_size=CHUNK_SIZE,
        causal=True,
    )

    q_blocks = (seq_len + BSA_BLOCK_SIZE - 1) // BSA_BLOCK_SIZE
    k_blocks = (seq_len + BSA_BLOCK_SIZE - 1) // BSA_BLOCK_SIZE
    mask_api_valid = mask_api[:, :, :q_blocks, :k_blocks]

    causal_mask = torch.tril(torch.ones(q_blocks, k_blocks, device=device, dtype=torch.bool))
    total_api = causal_mask.sum().item() * batch_size * num_heads
    selected_api = (mask_api_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
    density_api = selected_api / total_api

    # ========== 三阶段 KV Chunking ==========
    k_num_to_pad = ((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE - seq_len
    q_num_to_pad = ((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE - seq_len
    q_chunk_num = (seq_len + q_num_to_pad) // CHUNK_SIZE
    kv_chunk_num = (seq_len + k_num_to_pad) // CHUNK_SIZE

    k_block_num = (seq_len + k_num_to_pad) // BSA_BLOCK_SIZE
    q_block_num = (seq_len + q_num_to_pad) // BSA_BLOCK_SIZE

    reshaped_chunk_size = CHUNK_SIZE // STRIDE
    reshaped_block_size = BSA_BLOCK_SIZE // STRIDE
    k_reshaped_seq_len = (seq_len + k_num_to_pad) // STRIDE
    k_reshaped_num_to_pad = k_num_to_pad // STRIDE
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    kv_reshaped_chunk_size = CHUNK_SIZE // STRIDE

    if k_num_to_pad > 0:
        K_padded = torch.nn.functional.pad(K, (0, 0, 0, k_num_to_pad), value=0)
    else:
        K_padded = K

    if q_num_to_pad > 0:
        Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, q_num_to_pad), value=0)
    else:
        Q_padded = Q

    norm = 1.0
    scale = 1.4426950408889634 / math.sqrt(head_dim) / STRIDE / norm

    simple_mask_list = []

    for q_chunk_idx in range(q_chunk_num):
        q_start = q_chunk_idx * reshaped_chunk_size * STRIDE
        q_end = q_start + reshaped_chunk_size * STRIDE
        Q_chunk = Q_padded[:, :, q_start:q_end, :]

        chunk_start = (k_block_num - q_block_num) * reshaped_block_size + q_chunk_idx * reshaped_chunk_size
        chunk_end = chunk_start + reshaped_chunk_size

        m_chunks = []
        l_chunks = []
        attn_weights_chunks = []

        for kv_chunk_idx in range(kv_chunk_num):
            kv_start = kv_chunk_idx * CHUNK_SIZE
            kv_end = kv_start + CHUNK_SIZE
            K_chunk = K_padded[:, :, kv_start:kv_end, :]
            kv_offset_reshaped = kv_chunk_idx * kv_reshaped_chunk_size

            attn_weights_kv = flat_group_gemm_fuse_reshape(
                Q_chunk, K_chunk, STRIDE,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                is_causal=False,
            )
            attn_weights_chunks.append(attn_weights_kv)

            m_partial, l_partial = softmax_compute_partial_stats(
                attn_weights_kv,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                scale,
                chunk_start=chunk_start,
                kv_offset=kv_offset_reshaped,
                is_causal=True,
            )
            m_chunks.append(m_partial)
            l_chunks.append(l_partial)

        m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)

        attn_sum_per_kv = []
        for kv_chunk_idx, attn_weights_kv in enumerate(attn_weights_chunks):
            kv_offset_reshaped = kv_chunk_idx * kv_reshaped_chunk_size
            attn_sum_kv = softmax_normalize_and_block_sum(
                attn_weights_kv,
                m_global,
                l_global,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                chunk_start=chunk_start,
                real_q_len=k_reshaped_seq_len - k_reshaped_num_to_pad,
                scale=scale,
                kv_offset=kv_offset_reshaped,
                is_causal=True,
            )
            attn_sum_per_kv.append(attn_sum_kv)

        attn_sum_concat = torch.cat(attn_sum_per_kv, dim=-1)

        simple_mask = find_blocks_chunked(
            attn_sum_concat,
            current_index=k_block_num - q_block_num + q_chunk_idx * num_blocks_per_chunk,
            threshold=THRESHOLD,
            num_to_choose=None,
            decoding=False,
            mode="prefill",
            causal=True,
        )
        simple_mask_list.append(simple_mask)

    mask_kv_chunking = torch.cat(simple_mask_list, dim=2)

    # 应用与 xattn_estimate 相同的 causal mask 后处理 (xattn.py 第 1300-1306 行)
    mask_kv_chunking[:, :, -q_block_num:, -q_block_num:] = torch.where(
        torch.tril(torch.ones(q_block_num, q_block_num, dtype=bool, device=device), diagonal=0),
        mask_kv_chunking[:, :, -q_block_num:, -q_block_num:],
        False,
    )

    mask_kv_chunking_valid = mask_kv_chunking[:, :, :q_blocks, :k_blocks]
    selected_kv = (mask_kv_chunking_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
    density_kv = selected_kv / total_api

    mask_total = mask_api_valid.numel()
    mask_diff = (mask_api_valid != mask_kv_chunking_valid).sum().item()
    mask_diff_pct = 100 * mask_diff / mask_total

    return {
        "seq_len": seq_len,
        "stride": STRIDE,
        "threshold": THRESHOLD,
        "kv_chunks": kv_chunk_num,
        "density_api": density_api,
        "density_kv": density_kv,
        "density_diff": abs(density_api - density_kv),
        "mask_diff_pct": mask_diff_pct,
        "passed": abs(density_api - density_kv) < 1e-6 and mask_diff_pct < 0.01,
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "qkv_*.pt")))

    print("=" * 80)
    print("XAttention KV Chunking Alignment Test")
    print("=" * 80)
    print()

    results = []
    for f in files:
        fname = os.path.basename(f)
        print(f"Testing {fname}...", end=" ", flush=True)
        try:
            r = test_single_file(f)
            results.append(r)
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"{status} (seq_len={r['seq_len']}, kv_chunks={r['kv_chunks']})")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({"file": fname, "error": str(e)})

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print("| seq_len | stride | threshold | kv_chunks | density_api | density_kv | diff | mask_diff | status |")
    print("|---------|--------|-----------|-----------|-------------|------------|------|-----------|--------|")

    all_passed = True
    for r in results:
        if "error" in r:
            print(f"| ERROR | - | - | - | - | - | - | - | {r['error'][:20]} |")
            all_passed = False
        else:
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            print(f"| {r['seq_len']:>7} | {r['stride']:>6} | {r['threshold']:.2f} | {r['kv_chunks']:>9} | "
                  f"{r['density_api']:.6f} | {r['density_kv']:.6f} | {r['density_diff']:.6f} | "
                  f"{r['mask_diff_pct']:.4f}% | {status} |")

    print()
    if all_passed:
        print("test_xattn_kv_chunking_batch: ALL PASSED")
    else:
        print("test_xattn_kv_chunking_batch: SOME FAILED")


if __name__ == "__main__":
    main()
