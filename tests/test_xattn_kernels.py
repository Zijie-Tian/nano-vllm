"""
Test: XAttention Triton kernels

演示 XAttention 的两个核心 Triton kernel:
1. flat_group_gemm_fuse_reshape: 计算 stride reshape 后的 attention scores (反对角线求和)
2. softmax_fuse_block_sum: 对 attention scores 做 softmax 后按 block 求和

数据流:
  Q, K [batch, heads, seq_len, head_dim]
    ↓ flat_group_gemm_fuse_reshape
  attn_scores [batch, heads, seq_len/stride, seq_len/stride]
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

seq_len = 512       # Triton 要求 seq_len >= stride * BLOCK_M = 4 * 128 = 512
head_dim = 128
stride = 4
block_size = 128    # softmax block size (in reshaped space)
segment_size = 128  # Triton kernel 要求 segment_size >= block_size

# ============================================================
# 构造输入: 偶数位置=1, 奇数位置=2
# ============================================================

Q = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16).cuda()
K = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16).cuda()
for i in range(seq_len):
    if i % 2 == 0:
        Q[0, 0, i, :] = 1
        K[0, 0, i, :] = 1
    else:
        Q[0, 0, i, :] = 2
        K[0, 0, i, :] = 2

# ============================================================
# Step 1: flat_group_gemm_fuse_reshape
# ============================================================

attn_scores = flat_group_gemm_fuse_reshape(
    Q, K, stride,
    chunk_start=0,
    chunk_end=seq_len // stride,
    is_causal=False
)

# 验证: 反对角线求和
# 每个 stride x stride 块的反对角线: Q[奇]*K[偶] + Q[偶]*K[奇] = 2*1 + 1*2 = 4
# 反对角线有 stride/2 对，再乘以 head_dim
expected_gemm = (2*1 + 1*2) * (stride // 2) * head_dim
actual_gemm = attn_scores[0, 0, 0, 0].item()
assert actual_gemm == expected_gemm, f"flat_group_gemm: {actual_gemm} != {expected_gemm}"

# ============================================================
# Step 2: softmax_fuse_block_sum
# ============================================================

reshaped_len = seq_len // stride
scale = 1.4426950408889634  # log2(e) for exp2

block_sums = softmax_fuse_block_sum(
    attn_scores,
    block_size,
    segment_size,
    chunk_start=0,
    chunk_end=reshaped_len,
    real_q_len=reshaped_len,
    scale=scale,
    is_causal=False
)

# 验证: 每个 block 的 softmax 结果求和
# 所有 attn_scores 相同 → softmax 均匀分布 → block_sum = block_size^2 / reshaped_len
expected_sum = block_size * block_size / reshaped_len
actual_sum = block_sums[0, 0, 0, 0].item()
assert actual_sum == expected_sum, f"softmax_fuse_block_sum: {actual_sum} != {expected_sum}"

print("test_xattn_kernels: PASSED")
