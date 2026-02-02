"""
Test: 验证 xattn_estimate 与 KV chunking kernels 的一致性

使用真实 KV cache 数据，对比:
1. xattn_estimate (高层 API)
2. 三阶段 KV chunking (softmax_compute_partial_stats + merge + normalize)

三阶段 KV chunking 流程:
  1. softmax_compute_partial_stats: 计算每个 KV chunk 的 (m, l)
  2. merge_softmax_stats: Host 端合并所有 chunks 的 stats
  3. softmax_normalize_and_block_sum: 使用全局 stats 归一化

支持两种数据格式:
  1. offload 模式保存: {"query", "key", "stride", "threshold", "density", "layer_id"}
  2. GPU-only 模式保存: {"Q", "K", "chunk_size", "block_size", "stride", "threshold", "mask", "attn_sums", ...}

Usage:
    # 使用 offload 模式数据
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
        python tests/test_xattn_estimate_alignment.py

    # 使用 GPU-only 模式数据
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
        python tests/test_xattn_estimate_alignment.py --gpuonly
"""
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

import argparse
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
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--gpuonly", action="store_true", help="使用 GPU-only 模式保存的数据")
parser.add_argument("--data-file", type=str, default=None, help="数据文件路径")
parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 CHUNK_SIZE (用于测试不同分块大小)")
args = parser.parse_args()

# ============================================================
# 参数配置
# ============================================================
if args.gpuonly:
    DATA_FILE = args.data_file or "/home/zijie/Code/nano-vllm/results/mask_alignment/gpuonly_layer0.pt"
else:
    DATA_FILE = args.data_file or "/home/zijie/Code/nano-vllm/results/kvcache/qkv_32485.pt"

device = "cuda"

# ============================================================
# Step 1: 加载真实数据
# ============================================================
print("=" * 60)
print("Step 1: 加载真实 KV cache 数据")
print("=" * 60)

data = torch.load(DATA_FILE, map_location="cpu")

# 检测数据格式并加载
if "Q" in data:
    # GPU-only 模式保存的格式
    print(f"[INFO] 检测到 GPU-only 模式数据格式")
    Q = data["Q"].to(device)
    K = data["K"].to(device)
    BSA_BLOCK_SIZE = data.get("block_size", 128)
    CHUNK_SIZE = data.get("chunk_size", 4096)
    STRIDE = data.get("stride", 8)
    THRESHOLD = data.get("threshold", 0.9)
    if isinstance(THRESHOLD, torch.Tensor):
        THRESHOLD = THRESHOLD.item()
    # GPU-only 模式保存了 mask 和 attn_sums，可以用于验证
    saved_mask = data.get("mask", None)
    saved_attn_sums = data.get("attn_sums", None)
    saved_density = None  # GPU-only 模式没有保存 density
    layer_id = 0  # GPU-only 只保存 layer 0
else:
    # offload 模式保存的格式
    print(f"[INFO] 检测到 offload 模式数据格式")
    Q = data["query"].to(device)
    K = data["key"].to(device)
    BSA_BLOCK_SIZE = 128
    CHUNK_SIZE = 4096
    STRIDE = data["stride"]
    THRESHOLD = data["threshold"]
    if isinstance(THRESHOLD, torch.Tensor):
        THRESHOLD = THRESHOLD[0].item()
    saved_mask = None
    saved_attn_sums = None
    saved_density = data.get("density", None)
    layer_id = data.get("layer_id", 0)

batch_size, num_heads, seq_len, head_dim = Q.shape

# 命令行覆盖 CHUNK_SIZE
if args.chunk_size is not None:
    CHUNK_SIZE = args.chunk_size
    print(f"[INFO] 使用命令行指定的 CHUNK_SIZE={CHUNK_SIZE}")

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
if saved_density is not None:
    print(f"Data layer_id: {layer_id}, saved density: {saved_density:.4f}")
else:
    print(f"Data layer_id: {layer_id}")
print(f"使用参数: STRIDE={STRIDE}, THRESHOLD={THRESHOLD}, CHUNK_SIZE={CHUNK_SIZE}, BSA_BLOCK_SIZE={BSA_BLOCK_SIZE}")
print()

# ============================================================
# Step 2: 使用 xattn_estimate 高层 API
# ============================================================
print("=" * 60)
print("Step 2: 调用 xattn_estimate (高层 API)")
print("=" * 60)

attn_sums_api, mask_api = xattn_estimate(
    Q, K,
    block_size=BSA_BLOCK_SIZE,
    stride=STRIDE,
    threshold=THRESHOLD,
    chunk_size=CHUNK_SIZE,
    causal=True,
)

# 裁剪到有效区域
q_blocks = (seq_len + BSA_BLOCK_SIZE - 1) // BSA_BLOCK_SIZE
k_blocks = (seq_len + BSA_BLOCK_SIZE - 1) // BSA_BLOCK_SIZE
mask_api_valid = mask_api[:, :, :q_blocks, :k_blocks]

# 计算 density (causal)
causal_mask = torch.tril(torch.ones(q_blocks, k_blocks, device=device, dtype=torch.bool))
total_api = causal_mask.sum().item() * batch_size * num_heads
selected_api = (mask_api_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
density_api = selected_api / total_api

print(f"mask_api shape (padded): {mask_api.shape}")
print(f"mask_api_valid shape:    {mask_api_valid.shape}")
print(f"[xattn_estimate] density: {density_api:.6f} (selected={selected_api}, total={total_api})")
print()

# ============================================================
# Step 3: 三阶段 KV Chunking
# ============================================================
print("=" * 60)
print("Step 3: 三阶段 KV Chunking")
print("=" * 60)
print("  1) 每个 KV chunk 计算 partial stats")
print("  2) Host 端合并 stats")
print("  3) 使用全局 stats 归一化并计算 block sums")
print()

# 计算 padding 参数
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

print(f"seq_len: {seq_len}, q_chunk_num: {q_chunk_num}, kv_chunk_num: {kv_chunk_num}")
print()

# Padding
if k_num_to_pad > 0:
    K_padded = torch.nn.functional.pad(K, (0, 0, 0, k_num_to_pad), value=0)
else:
    K_padded = K

if q_num_to_pad > 0:
    Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, q_num_to_pad), value=0)
else:
    Q_padded = Q

# Softmax scale
norm = 1.0
scale = 1.4426950408889634 / math.sqrt(head_dim) / STRIDE / norm

simple_mask_list = []

for q_chunk_idx in range(q_chunk_num):
    q_start = q_chunk_idx * reshaped_chunk_size * STRIDE
    q_end = q_start + reshaped_chunk_size * STRIDE
    Q_chunk = Q_padded[:, :, q_start:q_end, :]

    chunk_start = (k_block_num - q_block_num) * reshaped_block_size + q_chunk_idx * reshaped_chunk_size
    chunk_end = chunk_start + reshaped_chunk_size

    # 阶段 1: 每个 KV chunk 计算 partial stats 和 raw scores
    m_chunks = []
    l_chunks = []
    attn_weights_chunks = []

    for kv_chunk_idx in range(kv_chunk_num):
        kv_start = kv_chunk_idx * CHUNK_SIZE
        kv_end = kv_start + CHUNK_SIZE
        K_chunk = K_padded[:, :, kv_start:kv_end, :]

        # KV offset in reshaped space
        kv_offset_reshaped = kv_chunk_idx * kv_reshaped_chunk_size

        # 计算 raw attention scores
        attn_weights_kv = flat_group_gemm_fuse_reshape(
            Q_chunk, K_chunk, STRIDE,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_causal=False,  # K 不完整，不能在这里用 causal
        )
        attn_weights_chunks.append(attn_weights_kv)

        # 计算 partial stats (带 causal mask)
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

    # 阶段 2: Host 端合并 stats
    m_global, l_global = merge_softmax_stats(m_chunks, l_chunks)

    # 阶段 3: 使用全局 stats 归一化并计算 block sums
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

    # 拼接各 KV chunk 的 block sums
    attn_sum_concat = torch.cat(attn_sum_per_kv, dim=-1)

    # 选择 blocks
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

    print(f"  Q chunk {q_chunk_idx}: merged {kv_chunk_num} KV chunks, attn_sum shape={attn_sum_concat.shape}")

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

print()
print(f"[KV chunking] density: {density_kv:.6f} (selected={selected_kv}, total={total_api})")
print()

# ============================================================
# Step 4: 对比结果
# ============================================================
print("=" * 60)
print("Step 4: 对比结果")
print("=" * 60)
print()

mask_total = mask_api_valid.numel()
mask_diff = (mask_api_valid != mask_kv_chunking_valid).sum().item()

print("| 方法 | density | 与 API 差异 | Mask 差异 |")
print("|------|---------|-------------|-----------|")
print(f"| xattn_estimate API | {density_api:.6f} | - | - |")
print(f"| KV chunking | {density_kv:.6f} | {abs(density_api - density_kv):.6f} | {100*mask_diff/mask_total:.4f}% |")
print()

passed = abs(density_api - density_kv) < 1e-6 and mask_diff / mask_total < 0.001

# ============================================================
# Step 5: 与 GPU-only 保存的数据对比 (如果有)
# ============================================================
if saved_mask is not None or saved_attn_sums is not None:
    print("=" * 60)
    print("Step 5: 与 GPU-only 保存的数据对比")
    print("=" * 60)
    print()

    if saved_mask is not None:
        saved_mask_gpu = saved_mask.to(device)
        # 比较 mask
        mask_saved_diff = (mask_api_valid != saved_mask_gpu).sum().item()
        mask_saved_total = saved_mask_gpu.numel()
        print(f"| xattn_estimate vs GPU-only saved mask | 差异 blocks: {mask_saved_diff} / {mask_saved_total} ({100*mask_saved_diff/mask_saved_total:.4f}%) |")

        if mask_saved_diff == 0:
            print("✅ mask 与 GPU-only 保存完全一致")
        else:
            print("❌ mask 与 GPU-only 保存存在差异")
            passed = False

    if saved_attn_sums is not None:
        saved_attn_sums_gpu = saved_attn_sums.to(device)
        # 需要从 xattn_estimate 获取 attn_sums
        # 重新调用一次获取 attn_sums
        attn_sums_check, _ = xattn_estimate(
            Q, K,
            block_size=BSA_BLOCK_SIZE,
            stride=STRIDE,
            threshold=THRESHOLD,
            chunk_size=CHUNK_SIZE,
            causal=True,
        )
        attn_sums_check_valid = attn_sums_check[:, :, :q_blocks, :k_blocks]

        max_diff = (attn_sums_check_valid - saved_attn_sums_gpu).abs().max().item()
        mean_diff = (attn_sums_check_valid - saved_attn_sums_gpu).abs().mean().item()
        print(f"| xattn_estimate vs GPU-only saved attn_sums | max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e} |")

        if max_diff < 1e-5:
            print("✅ attn_sums 与 GPU-only 保存一致")
        else:
            print("❌ attn_sums 与 GPU-only 保存存在差异")
            passed = False

    print()

if passed:
    print("test_xattn_estimate_alignment: PASSED")
else:
    print("test_xattn_estimate_alignment: FAILED")
