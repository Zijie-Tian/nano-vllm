"""
Test: 验证 xattn_estimate 与底层 kernel 调用的一致性

使用真实 KV cache 数据，分别调用:
1. xattn_estimate (高层 API)
2. flat_group_gemm_fuse_reshape + softmax_fuse_block_sum + find_blocks_chunked (底层 kernels)

底层 kernels 按 Q 分 chunk，与 xattn_estimate 内部逻辑一致，减少峰值内存占用。

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
# Step 3: 使用底层 kernels 手动计算 (按 Q 分 chunk)
# ============================================================
print("=" * 60)
print("Step 3: 使用底层 kernels 手动计算 (按 Q 分 chunk)")
print("=" * 60)

# 3.1 计算 padding 参数 (与 xattn_estimate 内部一致)
k_num_to_pad = ((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE - seq_len
q_num_to_pad = ((seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE - seq_len
k_chunk_num = (seq_len + k_num_to_pad) // CHUNK_SIZE
q_chunk_num = (seq_len + q_num_to_pad) // CHUNK_SIZE

k_block_num = (seq_len + k_num_to_pad) // BSA_BLOCK_SIZE
q_block_num = (seq_len + q_num_to_pad) // BSA_BLOCK_SIZE

reshaped_chunk_size = CHUNK_SIZE // STRIDE
reshaped_block_size = BSA_BLOCK_SIZE // STRIDE
k_reshaped_seq_len = (seq_len + k_num_to_pad) // STRIDE
k_reshaped_num_to_pad = k_num_to_pad // STRIDE
num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size

print(f"原始 seq_len: {seq_len}")
print(f"q_chunk_num: {q_chunk_num}, k_chunk_num: {k_chunk_num}")
print(f"q_block_num: {q_block_num}, k_block_num: {k_block_num}")
print(f"reshaped_chunk_size: {reshaped_chunk_size}, reshaped_block_size: {reshaped_block_size}")
print(f"num_blocks_per_chunk: {num_blocks_per_chunk}")
print()

# 3.2 Padding
if k_num_to_pad > 0:
    K_padded = torch.nn.functional.pad(K, (0, 0, 0, k_num_to_pad), value=0)
else:
    K_padded = K

if q_num_to_pad > 0:
    Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, q_num_to_pad), value=0)
else:
    Q_padded = Q

print(f"Q_padded shape: {Q_padded.shape}")
print(f"K_padded shape: {K_padded.shape}")
print()

# 3.3 按 Q chunk 处理 (与 xattn_estimate 内部逻辑一致)
norm = 1.0
scale = 1.4426950408889634 / math.sqrt(head_dim) / STRIDE / norm

simple_mask_list = []

print(f"按 Q 分 {q_chunk_num} 个 chunk 处理...")
for chunk_idx in range(q_chunk_num):
    # 提取当前 Q chunk (与 xattn_estimate line 811-816 一致)
    q_start = chunk_idx * reshaped_chunk_size * STRIDE
    q_end = q_start + reshaped_chunk_size * STRIDE
    Q_chunk = Q_padded[:, :, q_start:q_end, :]

    # 计算 chunk_start/chunk_end (与 xattn_estimate line 819-820 一致)
    chunk_start = (k_block_num - q_block_num) * reshaped_block_size + chunk_idx * reshaped_chunk_size
    chunk_end = chunk_start + reshaped_chunk_size

    # flat_group_gemm_fuse_reshape (与 xattn_estimate line 810-822 一致)
    attn_weights_slice = flat_group_gemm_fuse_reshape(
        Q_chunk, K_padded, STRIDE,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        is_causal=True,
    )

    # softmax_fuse_block_sum (与 xattn_estimate line 827-836 一致)
    attn_sum = softmax_fuse_block_sum(
        attn_weights_slice,
        reshaped_block_size,
        min(4096, reshaped_block_size),
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        real_q_len=k_reshaped_seq_len - k_reshaped_num_to_pad,
        scale=scale,
        is_causal=True,
    )

    # find_blocks_chunked (与 xattn_estimate line 887-895 一致)
    simple_mask = find_blocks_chunked(
        attn_sum,
        current_index=k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
        threshold=THRESHOLD,
        num_to_choose=None,
        decoding=False,
        mode="prefill",
        causal=True,
    )

    simple_mask_list.append(simple_mask)
    print(f"  Chunk {chunk_idx}: Q[{q_start}:{q_end}], attn shape={attn_weights_slice.shape}, mask shape={simple_mask.shape}")

# 3.4 合并所有 chunks 的 mask (与 xattn_estimate line 901-905 一致)
mask_manual = torch.cat(simple_mask_list, dim=2)
print(f"\n合并后 mask_manual shape: {mask_manual.shape}")

# 裁剪到有效区域
mask_manual_valid = mask_manual[:, :, :q_blocks, :k_blocks]
print(f"mask_manual_valid shape: {mask_manual_valid.shape}")

# 计算 density
selected_manual = (mask_manual_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
total_manual = total_api
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
mask_diff_ratio = mask_diff / mask_total
print(f"Mask 不同的元素数: {mask_diff} / {mask_total} ({100*mask_diff_ratio:.4f}%)")
print()

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
    print("⚠️ 与保存的 density 有差异，可能是参数不同")
