"""
Test: XAttention Triton kernels

演示 XAttention 的两个核心 Triton kernel:
1. flat_group_gemm_fuse_reshape: 计算 stride reshape 后的 attention scores (反对角线求和)
2. softmax_fuse_block_sum: 对 attention scores 做 softmax 后按 block 求和

数据流:
  Q [batch, heads, q_len, head_dim]
  K [batch, heads, kv_len, head_dim]
    ↓ flat_group_gemm_fuse_reshape
  attn_scores [batch, heads, q_len/stride, kv_len/stride]
    ↓ softmax_fuse_block_sum
  block_sums [batch, heads, q_blocks, k_blocks]
"""
import torch
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")
from nanovllm.ops.xattn import flat_group_gemm_fuse_reshape, softmax_fuse_block_sum

# ============================================================
# 参数配置
# ============================================================

# Triton 约束: q_len >= stride * BLOCK_M, kv_len >= stride * BLOCK_N
# A100: BLOCK_M = BLOCK_N = 128, 所以 min = 4 * 128 = 512
# RTX 3090: BLOCK_M = BLOCK_N = 64, 所以 min = 4 * 64 = 256
q_len = 512
kv_len = 2048
head_dim = 128
stride = 4
block_size = 128    # softmax block size (in reshaped space)
segment_size = 128  # Triton kernel 要求 segment_size >= block_size

# ============================================================
# 构造输入: 偶数位置=1, 奇数位置=2
# ============================================================

Q = torch.zeros(1, 1, q_len, head_dim, dtype=torch.bfloat16).cuda()
K = torch.zeros(1, 1, kv_len, head_dim, dtype=torch.bfloat16).cuda()

for i in range(q_len):
    if i % 2 == 0:
        Q[0, 0, i, :] = 1
    else:
        Q[0, 0, i, :] = 2

for i in range(kv_len):
    if i % 2 == 0:
        K[0, 0, i, :] = 1
    else:
        K[0, 0, i, :] = 2

# ============================================================
# Step 1: flat_group_gemm_fuse_reshape (chunked along K)
# ============================================================

q_reshaped_len = q_len // stride   # 128
kv_reshaped_len = kv_len // stride  # 512

# 将 K 沿着长度维度分成多个 chunk
k_chunk_size = 512  # 每个 chunk 512 tokens
num_k_chunks = kv_len // k_chunk_size  # 4 chunks

attn_scores_list = []
for k_chunk_idx in range(num_k_chunks):
    k_start = k_chunk_idx * k_chunk_size
    k_end = k_start + k_chunk_size
    K_chunk = K[:, :, k_start:k_end, :]  # [1, 1, k_chunk_size, head_dim]

    # 对每个 K chunk 调用 flat_group_gemm_fuse_reshape
    # 输出: [batch, heads, q_len/stride, k_chunk_size/stride]
    attn_chunk = flat_group_gemm_fuse_reshape(
        Q, K_chunk, stride,
        chunk_start=0,
        chunk_end=q_reshaped_len,
        is_causal=False
    )
    attn_scores_list.append(attn_chunk)

# 拼接所有 K chunks 的结果
# 每个 chunk: [1, 1, q_reshaped_len, k_chunk_size/stride]
# 拼接后: [1, 1, q_reshaped_len, kv_reshaped_len]
attn_scores = torch.cat(attn_scores_list, dim=-1)

# 验证 shape: [batch, heads, q_len/stride, kv_len/stride]
assert attn_scores.shape == (1, 1, q_reshaped_len, kv_reshaped_len), \
    f"shape mismatch: {attn_scores.shape} != (1, 1, {q_reshaped_len}, {kv_reshaped_len})"

# 验证: 反对角线求和
# 每个 stride x stride 块的反对角线: Q[奇]*K[偶] + Q[偶]*K[奇] = 2*1 + 1*2 = 4
# 反对角线有 stride/2 对，再乘以 head_dim
expected_gemm = (2*1 + 1*2) * (stride // 2) * head_dim
actual_gemm = attn_scores[0, 0, 0, 0].item()
assert actual_gemm == expected_gemm, f"flat_group_gemm: {actual_gemm} != {expected_gemm}"

# ============================================================
# Step 2: softmax_fuse_block_sum
# ============================================================

scale = 1.4426950408889634  # log2(e) for exp2

block_sums = softmax_fuse_block_sum(
    attn_scores,
    block_size,
    segment_size,
    chunk_start=0,
    chunk_end=q_reshaped_len,
    real_q_len=q_reshaped_len,
    scale=scale,
    is_causal=False
)

# 验证 shape: [batch, heads, q_blocks, k_blocks]
q_blocks = q_reshaped_len // block_size  # 128 / 128 = 1
k_blocks = kv_reshaped_len // block_size  # 512 / 128 = 4
assert block_sums.shape == (1, 1, q_blocks, k_blocks), \
    f"shape mismatch: {block_sums.shape} != (1, 1, {q_blocks}, {k_blocks})"

# 验证: 每个 block 的 softmax 结果求和
# 所有 attn_scores 相同 → softmax 均匀分布
# 每行对一个 K block 的贡献 = block_size / kv_reshaped_len
# 每个 Q block 有 block_size 行
# block_sum = block_size * (block_size / kv_reshaped_len)
expected_sum = block_size * block_size / kv_reshaped_len
actual_sum = block_sums[0, 0, 0, 0].item()
assert actual_sum == expected_sum, f"softmax_fuse_block_sum: {actual_sum} != {expected_sum}"

print("test_xattn_kernels: PASSED")
