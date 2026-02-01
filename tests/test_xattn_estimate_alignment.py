"""
Test: 验证 xattn_estimate 与 KV chunking kernels 的一致性

使用真实 KV cache 数据，对比:
1. xattn_estimate (高层 API)
2. 三阶段 KV chunking (softmax_compute_partial_stats + merge + normalize)

三阶段 KV chunking 流程:
  1. softmax_compute_partial_stats: 计算每个 KV chunk 的 (m, l)
  2. merge_softmax_stats: Host 端合并所有 chunks 的 stats
  3. softmax_normalize_and_block_sum: 使用全局 stats 归一化

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
        python tests/test_xattn_estimate_alignment.py
"""
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

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
DATA_FILE = "/home/zijie/Code/nano-vllm/results/kvcache/qkv_32485.pt"
BSA_BLOCK_SIZE = 128
CHUNK_SIZE = 16384  # xattn_estimate 默认值
USE_SAVED_PARAMS = True  # 设为 False 则使用默认值

device = "cuda"

# ============================================================
# Step 1: 加载真实数据
# ============================================================
print("=" * 60)
print("Step 1: 加载真实 KV cache 数据")
print("=" * 60)

data = torch.load(DATA_FILE, map_location="cpu")
Q = data["query"].to(device)  # [1, 32, seq_len, 128]
K = data["key"].to(device)    # [1, 32, seq_len, 128]

batch_size, num_heads, seq_len, head_dim = Q.shape

# 从保存的数据中读取参数
if USE_SAVED_PARAMS:
    STRIDE = data["stride"]
    THRESHOLD = data["threshold"][0].item() if isinstance(data["threshold"], torch.Tensor) else data["threshold"]
else:
    STRIDE = 8
    THRESHOLD = 0.9

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Data layer_id: {data['layer_id']}, saved density: {data['density']:.4f}")
print(f"使用参数: STRIDE={STRIDE}, THRESHOLD={THRESHOLD}, CHUNK_SIZE={CHUNK_SIZE}")
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

if abs(density_api - density_kv) < 1e-6 and mask_diff / mask_total < 0.001:
    print("test_xattn_estimate_alignment: PASSED")
else:
    print("test_xattn_estimate_alignment: FAILED")
