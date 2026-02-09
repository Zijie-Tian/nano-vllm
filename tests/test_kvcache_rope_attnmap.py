"""
Test: Load KVCache-RoPE data and compute post-RoPE attention map.

Loads cached Q/K tensors, computes full attention map per head.
- Short sequences (< 64k): full resolution, no grid.
- Long sequences (>= 64k): 128x128 grid block average, grid scales with length.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
        python tests/test_kvcache_rope_attnmap.py
"""

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

MODEL = "glm-4-9b"
LENGTH = "512k"
LAYER = 5
DATA_PATH = Path(__file__).parent.parent / f"results/kvcache-rope/{MODEL}/{LENGTH}/layer_{LAYER:02d}.pt"
OUT_DIR = Path(__file__).parent.parent / f"results/kvcache-rope-ret/{MODEL}/{LENGTH}"
DEVICE = "cuda"
CHUNK_SIZE = 4096
VIS_SIZE = 512  # for plotting only (full-res path)

# ============================================================
# Auto grid size based on sequence length
# ============================================================

def get_grid_size(seq_len):
    """No grid for < 64k; grid scales up for longer sequences."""
    if seq_len < 60000:
        return 1        # full resolution
    elif seq_len < 120000:
        return 128      # 64k
    elif seq_len < 240000:
        return 256      # 128k
    else:
        return 512      # 256k+

# ============================================================
# Full resolution path (grid_size == 1)
# ============================================================

def compute_head_full(post_q, post_k, h, kv_idx, seq_len, scale, device, chunk_size):
    """Compute full [S, S] attention map, chunked on GPU, stored on CPU."""
    k_h = post_k[:, kv_idx, :].to(device=device, dtype=torch.float32)
    attn_rows = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        q_chunk = post_q[start:end, h, :].to(device=device, dtype=torch.float32)
        scores = torch.mm(q_chunk, k_h.t()) * scale

        row_idx = torch.arange(start, end, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        scores.masked_fill_(col_idx > row_idx, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn_rows.append(attn.cpu())
        del scores, attn, q_chunk

    del k_h
    torch.cuda.empty_cache()
    return torch.cat(attn_rows, dim=0)  # [S, S]

# ============================================================
# Grid path (grid_size > 1)
# ============================================================

def compute_head_gridded(post_q, post_k, h, kv_idx, seq_len, scale,
                         device, chunk_size, grid_size):
    """Compute attention, stream-downsample with fixed grid_size blocks.

    Column pooling is done ON GPU to avoid transferring full [C, S] tensors to CPU.
    Only the small [C, n_grid] result is transferred per chunk.
    """
    k_h = post_k[:, kv_idx, :].to(device=device, dtype=torch.float32)
    n_grid = math.ceil(seq_len / grid_size)
    full_blocks = seq_len // grid_size
    remainder = seq_len % grid_size

    attn_ds = torch.zeros(n_grid, n_grid, dtype=torch.float64)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        C = end - start
        q_chunk = post_q[start:end, h, :].to(device=device, dtype=torch.float32)
        scores = torch.mm(q_chunk, k_h.t()) * scale

        row_idx = torch.arange(start, end, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        scores.masked_fill_(col_idx > row_idx, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [C, S] float32 on GPU
        del scores, q_chunk

        # Column pooling ON GPU: [C, S] -> [C, n_grid]
        col_pooled = torch.zeros(C, n_grid, device=device, dtype=torch.float32)
        if full_blocks > 0:
            col_pooled[:, :full_blocks] = attn[:, :full_blocks * grid_size].reshape(
                C, full_blocks, grid_size
            ).mean(dim=2)
        if remainder > 0:
            col_pooled[:, full_blocks] = attn[:, full_blocks * grid_size:].mean(dim=1)
        del attn

        # Transfer only small [C, n_grid] to CPU
        col_pooled_cpu = col_pooled.cpu().double()
        del col_pooled

        # Row accumulation: index_add
        ri_indices = torch.arange(start, end) // grid_size
        attn_ds.index_add_(0, ri_indices, col_pooled_cpu)
        del col_pooled_cpu

    # Average each grid row by bin size
    row_counts = torch.full((n_grid,), grid_size, dtype=torch.float64)
    if remainder > 0:
        row_counts[-1] = remainder
    attn_ds /= row_counts.unsqueeze(1)

    del k_h
    torch.cuda.empty_cache()
    return attn_ds.float()

# ============================================================
# Main
# ============================================================

assert DATA_PATH.exists(), f"Data not found: {DATA_PATH}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = torch.load(DATA_PATH, map_location="cpu")
post_q = data["post_rope_q"]  # [S, H, D]
post_k = data["post_rope_k"]  # [S, Hkv, D]
positions = data["positions"]

seq_len, num_heads, head_dim = post_q.shape
num_kv_heads = post_k.shape[1]
num_groups = num_heads // num_kv_heads
scale = 1.0 / math.sqrt(head_dim)

grid_size = get_grid_size(seq_len)
use_grid = grid_size > 1
n_grid = math.ceil(seq_len / grid_size) if use_grid else seq_len

print(f"Model: {MODEL}, Layer: {LAYER}, Seq length: {seq_len}")
print(f"  Q: {post_q.shape}  (heads={num_heads}, head_dim={head_dim})")
print(f"  K: {post_k.shape}  (kv_heads={num_kv_heads}, GQA groups={num_groups})")
if use_grid:
    print(f"  Mode: GRID  grid_size={grid_size}  output={n_grid}x{n_grid}")
else:
    print(f"  Mode: FULL  output={seq_len}x{seq_len}  ({seq_len*seq_len*4/1e9:.2f} GB/head)")

# Compute per head
attn_list = []
for h in range(num_heads):
    kv_idx = h // num_groups
    if use_grid:
        attn_h = compute_head_gridded(
            post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE, grid_size
        )
    else:
        attn_h = compute_head_full(
            post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE
        )
    attn_list.append(attn_h)

    if (h + 1) % 8 == 0:
        row_sum = attn_h[-1].sum().item()
        print(f"  Computed heads {h-6}..{h+1}/{num_heads}  (last_row_sum={row_sum:.6f})")

attn_map = torch.stack(attn_list, dim=0)  # [H, n_grid, n_grid] or [H, S, S]
del attn_list
print(f"\nResult: {attn_map.shape}, {attn_map.element_size()*attn_map.nelement()/1e9:.2f} GB")

# ============================================================
# Visualization
# ============================================================

print("Generating figure...")

# For full-res path, downsample for plotting
if not use_grid and attn_map.shape[1] > VIS_SIZE:
    attn_vis = torch.nn.functional.adaptive_avg_pool2d(
        attn_map.unsqueeze(1), (VIS_SIZE, VIS_SIZE)
    ).squeeze(1)
    print(f"  Downsampled for plot: {attn_map.shape[1]}x{attn_map.shape[1]} -> {VIS_SIZE}x{VIS_SIZE}")
else:
    attn_vis = attn_map

attn_np = attn_vis.numpy()
del attn_vis

nrows = num_kv_heads
ncols = num_groups

fig = plt.figure(figsize=(ncols * 3.2, nrows * 3.2 + 1.5))

outer_gs = gridspec.GridSpec(
    nrows, 1, figure=fig,
    hspace=0.35, top=0.92, bottom=0.04, left=0.05, right=0.93,
)

for g in range(nrows):
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, ncols, subplot_spec=outer_gs[g], wspace=0.08)
    for j in range(ncols):
        h = g * ncols + j
        ax = fig.add_subplot(inner_gs[j])
        img = np.log10(attn_np[h] + 1e-8)
        ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest",
                  vmin=-6, vmax=0)
        ax.set_title(f"H{h}", fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"KV {g}", fontsize=10, fontweight="bold")

cbar_ax = fig.add_axes([0.94, 0.04, 0.015, 0.88])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=-6, vmax=0))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("log10(attention)", fontsize=10)

mode_str = f"grid={grid_size}x{grid_size} → {n_grid}x{n_grid}" if use_grid else "full resolution"
fig.suptitle(
    f"{MODEL}  Layer {LAYER}  ({LENGTH}, seq={seq_len})  —  Post-RoPE Attention Map\n"
    f"4 KV groups × 8 Q heads  |  {mode_str}",
    fontsize=13, fontweight="bold",
)

out_path = OUT_DIR / f"layer_{LAYER:02d}_attnmap.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

print("\ntest_kvcache_rope_attnmap: PASSED")
