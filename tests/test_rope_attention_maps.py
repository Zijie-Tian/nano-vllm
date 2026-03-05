"""
研究 RoPE 对稀疏 attention 的影响

生成四个 attention map:
1. 没有 RoPE 的 attention map (使用 pre_rope_q/k)
2. 有 RoPE 的 attention map (使用 post_rope_q/k)
3. RoPE 本身的 attention map (差异可视化)
4. 纯 RoPE attention map (合成数据，展示位置编码特性)

使用 kvcache-rope 数据
"""
import torch
import torch.nn.functional as F
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/zijie/Code/COMPASS")

from compass.src.Xattention import xattn_estimate

# ============================================================
# Configuration
# ============================================================
BLOCK_SIZE = 128
STRIDE = 8
THRESHOLD = 0.9
MODEL = "glm-4-9b"
LAYER = 5
SEQ_LEN = 4096  # 使用 4K 序列以便可视化
HEAD_IDX = 0    # 选择第一个 head 进行可视化

# ============================================================
# Load Data
# ============================================================
def load_data(data_path):
    """Load kvcache-rope data."""
    data = torch.load(data_path, map_location="cpu")
    return data

def prepare_qkv(data, seq_len, head_idx=0):
    """
    准备 Q/K/V 数据
    返回指定 head 的 pre-rope 和 post-rope Q/K
    """
    pre_q = data['pre_rope_q'][:seq_len]  # [seq_len, num_heads, head_dim]
    pre_k = data['pre_rope_k'][:seq_len]
    post_q = data['post_rope_q'][:seq_len]
    post_k = data['post_rope_k'][:seq_len]
    v = data['v'][:seq_len]

    # 对于 GQA, 需要复制 K 来匹配 Q 的 head 数
    num_heads = pre_q.shape[1]
    num_kv_heads = pre_k.shape[1]
    num_groups = num_heads // num_kv_heads

    # 选择指定 head 的 Q 和对应的 K (考虑 GQA)
    kv_head_idx = head_idx // num_groups

    # 提取数据: [seq_len, head_dim]
    pre_q_h = pre_q[:, head_idx, :]
    pre_k_h = pre_k[:, kv_head_idx, :]
    post_q_h = post_q[:, head_idx, :]
    post_k_h = post_k[:, kv_head_idx, :]
    v_h = v[:, kv_head_idx, :]

    # 转换为 [batch=1, heads=1, seq_len, head_dim]
    pre_q_h = pre_q_h.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    pre_k_h = pre_k_h.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    post_q_h = post_q_h.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    post_k_h = post_k_h.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    v_h = v_h.unsqueeze(0).unsqueeze(0).transpose(2, 3)

    return {
        'pre_q': pre_q_h.cuda().to(torch.bfloat16),
        'pre_k': pre_k_h.cuda().to(torch.bfloat16),
        'post_q': post_q_h.cuda().to(torch.bfloat16),
        'post_k': post_k_h.cuda().to(torch.bfloat16),
        'v': v_h.cuda().to(torch.bfloat16),
    }

# ============================================================
# Compute Attention Maps
# ============================================================
def compute_attention_map(q, k, block_size, stride, threshold):
    """使用 xattn_estimate 计算稀疏 attention map."""
    attn_sum, mask = xattn_estimate(
        q, k,
        block_size=block_size,
        stride=stride,
        threshold=threshold,
        use_triton=False,  # 使用 PyTorch 实现避免共享内存问题
        causal=True,
    )
    return attn_sum, mask

def compute_dense_attention(q, k, scale=None):
    """计算 dense attention 用于对比."""
    batch, heads, q_len, head_dim = q.shape

    # 默认 scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    causal_mask = torch.triu(torch.ones(q_len, q_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    return attn


def apply_rope(q, k, positions, head_dim):
    """
    应用 RoPE (Rotary Position Embedding).

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        positions: [seq_len] position indices
        head_dim: dimension per head

    Returns:
        q_rot, k_rot: RoPE 编码后的 Q 和 K
    """
    batch, heads, seq_len, _ = q.shape

    # 计算旋转频率 (theta)
    # theta_i = base^(-2i/d) for i in [0, d/2)
    base = 10000.0
    dim_indices = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    theta = base ** (-dim_indices / head_dim)  # [head_dim/2]

    # 计算旋转角度: angle = position * theta
    # positions: [seq_len], theta: [head_dim/2]
    angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # [seq_len, head_dim/2]

    # 计算 cos 和 sin
    cos = torch.cos(angles)  # [seq_len, head_dim/2]
    sin = torch.sin(angles)  # [seq_len, head_dim/2]

    # 扩展维度以匹配 q/k
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]

    # 将 q/k 分成两半 (x1, x2)
    q1, q2 = q[..., ::2], q[..., 1::2]  # 每半个都是 [batch, heads, seq_len, head_dim/2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # 应用旋转: [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
    q_rot = torch.stack([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)  # [batch, heads, seq_len, head_dim/2, 2]
    k_rot = torch.stack([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)

    # 将最后两维展开为 head_dim
    q_rot = q_rot.view(batch, heads, seq_len, head_dim)
    k_rot = k_rot.view(batch, heads, seq_len, head_dim)

    return q_rot, k_rot


def compute_pure_rope_attention_map(seq_len, head_dim, device='cuda'):
    """
    计算纯 RoPE attention map (使用单位向量展示位置编码特性).

    使用单位向量作为基础，只关注 RoPE 引入的相对位置影响。
    """
    # 创建单位向量 (所有维度都是 1/sqrt(head_dim) 以便归一化)
    q_raw = torch.ones(1, 1, seq_len, head_dim, device=device) / math.sqrt(head_dim)
    k_raw = torch.ones(1, 1, seq_len, head_dim, device=device) / math.sqrt(head_dim)

    # 创建位置索引
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)

    # 应用 RoPE
    q_rope, k_rope = apply_rope(q_raw, k_raw, positions, head_dim)

    # 计算 attention scores (不使用 softmax，直接看原始分数)
    scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))  # [1, 1, seq_len, seq_len]

    # 应用 causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax 归一化
    attn = F.softmax(scores, dim=-1)

    return attn, scores

# ============================================================
# Visualization
# ============================================================
def visualize_attention_maps(maps, titles, save_path):
    """可视化多个 attention map."""
    n_maps = len(maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(6*n_maps, 5))

    if n_maps == 1:
        axes = [axes]

    for ax, map_data, title in zip(axes, maps, titles):
        # 转换为 numpy
        if isinstance(map_data, torch.Tensor):
            map_np = map_data.cpu().float().numpy()
        else:
            map_np = map_data

        # 如果是 4D tensor [B, H, Q, K], 取第一个 batch 和 head
        if map_np.ndim == 4:
            map_np = map_np[0, 0]

        im = ax.imshow(map_np, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
    plt.close()

def visualize_difference(map1, map2, title, save_path):
    """可视化两个 attention map 的差异."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Map 1
    m1 = map1[0, 0].cpu().float().numpy() if isinstance(map1, torch.Tensor) else map1[0, 0]
    im1 = axes[0].imshow(m1, cmap='viridis', aspect='auto')
    axes[0].set_title('Without RoPE', fontsize=12)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])

    # Map 2
    m2 = map2[0, 0].cpu().float().numpy() if isinstance(map2, torch.Tensor) else map2[0, 0]
    im2 = axes[1].imshow(m2, cmap='viridis', aspect='auto')
    axes[1].set_title('With RoPE', fontsize=12)
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = m2 - m1
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[2].set_title('Difference (RoPE effect)', fontsize=12)
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved difference figure to: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RoPE Impact on Sparse Attention - Attention Map Analysis")
    print("=" * 60)

    # Load data
    DATA_PATH = f"/home/zijie/Code/COMPASS/results/kvcache-rope/{MODEL}/16k/layer_{LAYER:02d}.pt"
    print(f"\nLoading data from: {DATA_PATH}")
    data = load_data(DATA_PATH)

    # Prepare Q/K
    print(f"Preparing Q/K for head {HEAD_IDX}, seq_len={SEQ_LEN}")
    qkv = prepare_qkv(data, SEQ_LEN, HEAD_IDX)

    # Create output directory
    OUTPUT_DIR = "/home/zijie/Code/COMPASS/results/rope_attention_maps"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # 1. Compute Dense Attention Maps (for visualization)
    # ============================================================
    print("\n" + "=" * 60)
    print("1. Computing Dense Attention Maps")
    print("=" * 60)

    with torch.no_grad():
        # Pre-RoPE dense attention
        attn_pre_dense = compute_dense_attention(qkv['pre_q'], qkv['pre_k'])
        print(f"Pre-RoPE dense attention shape: {attn_pre_dense.shape}")

        # Post-RoPE dense attention
        attn_post_dense = compute_dense_attention(qkv['post_q'], qkv['post_k'])
        print(f"Post-RoPE dense attention shape: {attn_post_dense.shape}")

        # RoPE effect (difference)
        attn_rope_effect = attn_post_dense - attn_pre_dense
        print(f"RoPE effect range: [{attn_rope_effect.min():.4f}, {attn_rope_effect.max():.4f}]")

    # Visualize dense attention maps
    visualize_attention_maps(
        [attn_pre_dense[0, 0], attn_post_dense[0, 0], attn_rope_effect[0, 0]],
        ['Without RoPE', 'With RoPE', 'RoPE Effect (Difference)'],
        f"{OUTPUT_DIR}/dense_attention_maps_head{HEAD_IDX}.png"
    )

    # Visualize difference in detail
    visualize_difference(
        attn_pre_dense, attn_post_dense,
        f"Dense Attention - RoPE Impact (Head {HEAD_IDX})",
        f"{OUTPUT_DIR}/dense_attention_diff_head{HEAD_IDX}.png"
    )

    # ============================================================
    # 2. Compute Sparse Attention Maps
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Computing Sparse Attention Maps (XAttention)")
    print("=" * 60)

    with torch.no_grad():
        # Pre-RoPE sparse attention
        attn_sum_pre, mask_pre = compute_attention_map(
            qkv['pre_q'], qkv['pre_k'],
            BLOCK_SIZE, STRIDE, THRESHOLD
        )
        print(f"Pre-RoPE sparse attention sum shape: {attn_sum_pre.shape}")
        print(f"Pre-RoPE mask density: {mask_pre.float().mean():.4f}")

        # Post-RoPE sparse attention
        attn_sum_post, mask_post = compute_attention_map(
            qkv['post_q'], qkv['post_k'],
            BLOCK_SIZE, STRIDE, THRESHOLD
        )
        print(f"Post-RoPE sparse attention sum shape: {attn_sum_post.shape}")
        print(f"Post-RoPE mask density: {mask_post.float().mean():.4f}")

    # Visualize sparse attention masks
    visualize_attention_maps(
        [mask_pre[0, 0].float(), mask_post[0, 0].float()],
        ['Sparse Mask (No RoPE)', 'Sparse Mask (With RoPE)'],
        f"{OUTPUT_DIR}/sparse_masks_head{HEAD_IDX}.png"
    )

    # Visualize sparse attention sums
    visualize_attention_maps(
        [attn_sum_pre[0, 0], attn_sum_post[0, 0]],
        ['Attn Sum (No RoPE)', 'Attn Sum (With RoPE)'],
        f"{OUTPUT_DIR}/sparse_attention_sums_head{HEAD_IDX}.png"
    )

    # ============================================================
    # 3. Compute Mask Difference (RoPE impact on sparsity)
    # ============================================================
    print("\n" + "=" * 60)
    print("3. Analyzing RoPE Impact on Sparsity Pattern")
    print("=" * 60)

    mask_pre_bool = mask_pre[0, 0].cpu().numpy()
    mask_post_bool = mask_post[0, 0].cpu().numpy()

    # Mask agreement
    agreement = (mask_pre_bool == mask_post_bool).mean()
    print(f"Mask agreement: {agreement:.4f} ({agreement*100:.2f}%)")

    # IoU
    intersection = (mask_pre_bool & mask_post_bool).sum()
    union = (mask_pre_bool | mask_post_bool).sum()
    iou = intersection / union if union > 0 else 0
    print(f"Mask IoU: {iou:.4f}")

    # Blocks only in pre-RoPE
    only_pre = mask_pre_bool & ~mask_post_bool
    print(f"Blocks only in pre-RoPE: {only_pre.sum()}")

    # Blocks only in post-RoPE
    only_post = ~mask_pre_bool & mask_post_bool
    print(f"Blocks only in post-RoPE: {only_post.sum()}")

    # Visualize mask difference
    mask_diff = mask_post_bool.astype(int) - mask_pre_bool.astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mask_diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Sparse Mask Difference (Post - Pre RoPE)\nHead {HEAD_IDX}', fontsize=12)
    ax.set_xlabel('Key Block')
    ax.set_ylabel('Query Block')
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.set_ticklabels(['Only Pre', 'Same', 'Only Post'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mask_difference_head{HEAD_IDX}.png", dpi=150, bbox_inches='tight')
    print(f"Saved mask difference to: {OUTPUT_DIR}/mask_difference_head{HEAD_IDX}.png")
    plt.close()

    # ============================================================
    # 4. Pure RoPE Attention Map (Synthetic)
    # ============================================================
    print("\n" + "=" * 60)
    print("4. Computing Pure RoPE Attention Map (Synthetic)")
    print("=" * 60)

    # 使用较小的序列长度以便可视化 RoPE 的周期性
    PURE_ROPE_SEQ_LEN = 512
    head_dim = qkv['pre_q'].shape[-1]

    with torch.no_grad():
        pure_rope_attn, pure_rope_scores = compute_pure_rope_attention_map(
            PURE_ROPE_SEQ_LEN, head_dim, device='cuda'
        )
        print(f"Pure RoPE attention shape: {pure_rope_attn.shape}")
        print(f"Attention score range: [{pure_rope_scores.min():.4f}, {pure_rope_scores.max():.4f}]")

    # Visualize pure RoPE attention map
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pure RoPE attention (after softmax)
    attn_np = pure_rope_attn[0, 0].cpu().float().numpy()
    im1 = axes[0].imshow(attn_np, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Pure RoPE Attention Map\n(seq_len={PURE_ROPE_SEQ_LEN}, head_dim={head_dim})', fontsize=12)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])

    # Pure RoPE attention (raw scores before softmax)
    # Replace -inf with NaN for visualization
    scores_np = pure_rope_scores[0, 0].cpu().float().numpy()
    scores_np = np.where(np.isinf(scores_np), np.nan, scores_np)
    im2 = axes[1].imshow(scores_np, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    axes[1].set_title(f'Pure RoPE Raw Scores\n(before softmax, causal masked)', fontsize=12)
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pure_rope_attention_map.png", dpi=150, bbox_inches='tight')
    print(f"Saved pure RoPE figure to: {OUTPUT_DIR}/pure_rope_attention_map.png")
    plt.close()

    # Additional: Visualize attention for a single query position
    fig, ax = plt.subplots(figsize=(12, 4))
    q_idx = PURE_ROPE_SEQ_LEN // 2  # 选择中间位置
    attn_slice = attn_np[q_idx, :q_idx+1]  # 只显示 causal 部分

    ax.plot(range(q_idx+1), attn_slice, linewidth=1.5, color='steelblue')
    ax.fill_between(range(q_idx+1), attn_slice, alpha=0.3, color='steelblue')
    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=11)
    ax.set_title(f'Pure RoPE Attention Distribution for Query Position {q_idx}\n(showing relative position bias)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, q_idx)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pure_rope_attention_slice.png", dpi=150, bbox_inches='tight')
    print(f"Saved pure RoPE slice to: {OUTPUT_DIR}/pure_rope_attention_slice.png")
    plt.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  1. dense_attention_maps_head{HEAD_IDX}.png - Dense attention comparison")
    print(f"  2. dense_attention_diff_head{HEAD_IDX}.png - Dense attention difference")
    print(f"  3. sparse_masks_head{HEAD_IDX}.png - Sparse mask comparison")
    print(f"  4. sparse_attention_sums_head{HEAD_IDX}.png - Sparse attention sum comparison")
    print(f"  5. mask_difference_head{HEAD_IDX}.png - Mask difference visualization")
    print(f"  6. pure_rope_attention_map.png - Pure RoPE attention map (synthetic)")
    print(f"  7. pure_rope_attention_slice.png - Pure RoPE attention slice")
    print(f"\nMetrics:")
    print(f"  Mask agreement: {agreement:.4f}")
    print(f"  Mask IoU: {iou:.4f}")
    print("=" * 60)
