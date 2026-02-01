"""
Test: 验证 xattn_estimate 与底层 kernel 调用的一致性

使用真实 KV cache 数据，分别调用:
1. xattn_estimate (高层 API)
2. flat_group_gemm_fuse_reshape + softmax_fuse_block_sum + find_blocks_chunked (底层 kernels)

验证两种方式的 density 是否一致。

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
    softmax_fuse_block_sum,
    find_blocks_chunked,
    compute_sparsity,
)

# ============================================================
# 参数配置
# ============================================================
DATA_FILE = "/home/zijie/Code/nano-vllm/results/kvcache/qkv_32485.pt"
BSA_BLOCK_SIZE = 128
# STRIDE 和 THRESHOLD 从保存的数据中读取
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
print(f"使用参数: STRIDE={STRIDE}, THRESHOLD={THRESHOLD}")
print()

# ============================================================
# Step 2: 使用 xattn_estimate 高层 API
# ============================================================
print("=" * 60)
print("Step 2: 调用 xattn_estimate (高层 API)")
print("=" * 60)

# 使用与底层计算一致的 chunk_size (seq_len 对齐到 alignment)
alignment = STRIDE * 128
chunk_size_aligned = ((seq_len + alignment - 1) // alignment) * alignment

attn_sums_api, mask_api = xattn_estimate(
    Q, K,
    block_size=BSA_BLOCK_SIZE,
    stride=STRIDE,
    threshold=THRESHOLD,
    chunk_size=chunk_size_aligned,  # 保持一致
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
# Step 3: 使用底层 kernels 手动计算
# ============================================================
print("=" * 60)
print("Step 3: 使用底层 kernels 手动计算")
print("=" * 60)

# 3.1 Padding
BLOCK_M = 128
BLOCK_N = 128
alignment = STRIDE * BLOCK_M
k_alignment = STRIDE * BLOCK_N

padded_q_len = ((seq_len + alignment - 1) // alignment) * alignment
padded_k_len = ((seq_len + k_alignment - 1) // k_alignment) * k_alignment

print(f"原始 seq_len: {seq_len}")
print(f"Padded Q len: {padded_q_len}")
print(f"Padded K len: {padded_k_len}")

if padded_q_len != seq_len:
    Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, padded_q_len - seq_len), value=0)
else:
    Q_padded = Q

if padded_k_len != seq_len:
    K_padded = torch.nn.functional.pad(K, (0, 0, 0, padded_k_len - seq_len), value=0)
else:
    K_padded = K

print(f"Q_padded shape: {Q_padded.shape}")
print(f"K_padded shape: {K_padded.shape}")
print()

# 3.2 计算 reshaped 维度
q_reshaped_len = padded_q_len // STRIDE
k_reshaped_len = padded_k_len // STRIDE
reshaped_block_size = BSA_BLOCK_SIZE // STRIDE

q_block_num = padded_q_len // BSA_BLOCK_SIZE
k_block_num = padded_k_len // BSA_BLOCK_SIZE

print(f"q_reshaped_len: {q_reshaped_len}")
print(f"k_reshaped_len: {k_reshaped_len}")
print(f"reshaped_block_size: {reshaped_block_size}")
print(f"q_block_num: {q_block_num}, k_block_num: {k_block_num}")
print()

# 3.3 调用 flat_group_gemm_fuse_reshape
print("3.3 调用 flat_group_gemm_fuse_reshape...")
chunk_start = (k_block_num - q_block_num) * reshaped_block_size  # 对于 q_len=k_len, offset=0
chunk_end = chunk_start + q_reshaped_len

attn_scores = flat_group_gemm_fuse_reshape(
    Q_padded, K_padded, STRIDE,
    chunk_start=chunk_start,
    chunk_end=chunk_end,
    is_causal=True,
)
print(f"attn_scores shape: {attn_scores.shape}")
print()

# 3.4 调用 softmax_fuse_block_sum
print("3.4 调用 softmax_fuse_block_sum...")
norm = 1.0
scale = 1.4426950408889634 / math.sqrt(head_dim) / STRIDE / norm
segment_size = min(4096, reshaped_block_size)

# 计算 real_q_len (排除 padding)
k_reshaped_num_to_pad = (padded_k_len - seq_len) // STRIDE
real_q_len = k_reshaped_len - k_reshaped_num_to_pad

block_sums = softmax_fuse_block_sum(
    attn_scores,
    reshaped_block_size,
    segment_size,
    chunk_start=chunk_start,
    chunk_end=chunk_end,
    real_q_len=real_q_len,
    scale=scale,
    is_causal=True,
)
print(f"block_sums shape: {block_sums.shape}")
print()

# 3.5 调用 find_blocks_chunked
print("3.5 调用 find_blocks_chunked...")
mask_manual = find_blocks_chunked(
    block_sums,
    current_index=0,  # Q 从位置 0 开始 (因为 q_len = k_len)
    threshold=THRESHOLD,
    num_to_choose=None,
    decoding=False,
    mode="prefill",
    causal=True,
)

# 裁剪到有效区域
mask_manual_valid = mask_manual[:, :, :q_blocks, :k_blocks]
print(f"mask_manual shape (padded): {mask_manual.shape}")
print(f"mask_manual_valid shape:    {mask_manual_valid.shape}")

# 计算 density
selected_manual = (mask_manual_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
total_manual = total_api  # 相同的 total
density_manual = selected_manual / total_manual

print(f"[底层 kernels] density: {density_manual:.6f} (selected={selected_manual}, total={total_manual})")
print()

# ============================================================
# Step 4: 对比结果
# ============================================================
print("=" * 60)
print("Step 4: 对比结果")
print("=" * 60)

print(f"xattn_estimate density:  {density_api:.6f}")
print(f"底层 kernels density:    {density_manual:.6f}")
print(f"差异:                    {abs(density_api - density_manual):.6f}")
print()

# 对比 mask
mask_diff = (mask_api_valid != mask_manual_valid).sum().item()
mask_total = mask_api_valid.numel()
print(f"Mask 不同的元素数: {mask_diff} / {mask_total} ({100*mask_diff/mask_total:.4f}%)")
print()

mask_diff_ratio = mask_diff / mask_total
if abs(density_api - density_manual) < 1e-6 and mask_diff_ratio < 0.001:
    print("✅ xattn_estimate 与底层 kernels 对齐! (mask 差异 < 0.1%)")
elif abs(density_api - density_manual) < 0.01:
    print("⚠️ Density 基本一致，但 mask 有差异")
else:
    print("❌ Density 不一致，需要检查参数")

# ============================================================
# Step 5: 额外验证 - 与保存的 density 对比
# ============================================================
print()
print("=" * 60)
print("Step 5: 与保存的 density 对比")
print("=" * 60)
saved_density = data["density"]
print(f"保存的 density:          {saved_density:.6f}")
print(f"xattn_estimate density:  {density_api:.6f}")
print(f"差异:                    {abs(saved_density - density_api):.6f}")

if abs(saved_density - density_api) < 0.01:
    print("✅ 与保存的 density 基本一致!")
else:
    print("⚠️ 与保存的 density 有差异，可能是 threshold 不同")
