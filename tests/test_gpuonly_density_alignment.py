"""
Test: GPU-only density alignment verification

验证 xattn_bsa.py 中 GPU-only 路径的 density 计算是否与独立调用 xattn_estimate 一致。

流程：
1. 运行 GPU-only 推理，保存 Q, K, mask, attn_sums
2. 加载保存的数据，独立调用 xattn_estimate
3. 比较两者的 mask 和 density
"""
import torch
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

from nanovllm.ops.xattn import xattn_estimate

# ============================================================
# 参数配置
# ============================================================

DATA_PATH = "/home/zijie/Code/nano-vllm/results/mask_alignment/gpuonly_layer0.pt"

# ============================================================
# 加载保存的数据
# ============================================================

print(f"Loading data from {DATA_PATH}")
data = torch.load(DATA_PATH, weights_only=False)

Q = data["Q"].cuda()  # [1, num_heads, q_len, head_dim]
K = data["K"].cuda()  # [1, num_heads, k_len, head_dim]
chunk_size = data["chunk_size"]
block_size = data["block_size"]
stride = data["stride"]
threshold = data["threshold"]
mask_saved = data["mask"]  # [1, num_heads, valid_q_blocks, valid_k_blocks]
attn_sums_saved = data["attn_sums"]
q_len = data["q_len"]
k_len = data["k_len"]
valid_q_blocks = data["valid_q_blocks"]
valid_k_blocks = data["valid_k_blocks"]

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"q_len: {q_len}, k_len: {k_len}")
print(f"chunk_size: {chunk_size}, block_size: {block_size}, stride: {stride}, threshold: {threshold}")
print(f"valid_q_blocks: {valid_q_blocks}, valid_k_blocks: {valid_k_blocks}")
print(f"mask_saved shape: {mask_saved.shape}")

# ============================================================
# 独立调用 xattn_estimate
# ============================================================

print("\nCalling xattn_estimate independently...")
attn_sums_ext, mask_ext = xattn_estimate(
    Q, K,
    chunk_size=chunk_size,
    block_size=block_size,
    stride=stride,
    threshold=threshold,
    use_triton=True,
    causal=True,
)

# Trim to valid blocks
mask_ext_valid = mask_ext[:, :, :valid_q_blocks, :valid_k_blocks]
attn_sums_ext_valid = attn_sums_ext[:, :, :valid_q_blocks, :valid_k_blocks]

print(f"mask_ext shape: {mask_ext.shape}")
print(f"mask_ext_valid shape: {mask_ext_valid.shape}")

# ============================================================
# 比较 attn_sums
# ============================================================

print("\n" + "=" * 60)
print("Comparing attn_sums")
print("=" * 60)

attn_sums_saved_gpu = attn_sums_saved.cuda()
attn_diff = (attn_sums_ext_valid - attn_sums_saved_gpu).abs()
print(f"attn_sums max diff: {attn_diff.max().item():.6e}")
print(f"attn_sums mean diff: {attn_diff.mean().item():.6e}")

# Check if attn_sums match
attn_match = attn_diff.max().item() < 1e-4
print(f"attn_sums match: {attn_match}")

# ============================================================
# 比较 mask
# ============================================================

print("\n" + "=" * 60)
print("Comparing mask")
print("=" * 60)

mask_saved_gpu = mask_saved.cuda()
mask_match = (mask_ext_valid == mask_saved_gpu).all().item()
print(f"mask exact match: {mask_match}")

if not mask_match:
    diff_count = (mask_ext_valid != mask_saved_gpu).sum().item()
    total_count = mask_ext_valid.numel()
    print(f"mask diff count: {diff_count} / {total_count} ({diff_count/total_count*100:.2f}%)")

# ============================================================
# 计算 density
# ============================================================

print("\n" + "=" * 60)
print("Comparing density")
print("=" * 60)

# 计算 causal mask
q_offset_blocks = valid_k_blocks - valid_q_blocks
indices = torch.arange(valid_k_blocks, device=mask_ext_valid.device).unsqueeze(0)
q_indices = torch.arange(valid_q_blocks, device=mask_ext_valid.device).unsqueeze(1)
causal_mask = indices <= (q_indices + q_offset_blocks)

# Density from saved mask
total_saved = causal_mask.sum().item() * mask_saved_gpu.shape[0] * mask_saved_gpu.shape[1]
selected_saved = (mask_saved_gpu & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
density_saved = selected_saved / total_saved

# Density from external xattn_estimate
total_ext = causal_mask.sum().item() * mask_ext_valid.shape[0] * mask_ext_valid.shape[1]
selected_ext = (mask_ext_valid & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
density_ext = selected_ext / total_ext

print(f"Saved density:    {density_saved:.6f} (selected={selected_saved}, total={total_saved})")
print(f"External density: {density_ext:.6f} (selected={selected_ext}, total={total_ext})")
print(f"Density diff:     {abs(density_saved - density_ext):.6f}")

# ============================================================
# 结论
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

if attn_match and mask_match:
    print("✅ PASSED: GPU-only density matches external xattn_estimate")
else:
    print("❌ FAILED: Mismatch detected")
    if not attn_match:
        print("  - attn_sums mismatch")
    if not mask_match:
        print("  - mask mismatch")
