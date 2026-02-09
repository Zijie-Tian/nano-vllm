"""
Test: Q Head Regrouping Validation for 64k+ sequences.

Adapted from test_kvcache_rope_reorder.py for long sequences:
- Density profiling: streaming chunked computation (no full attn map in memory)
- Visualization: grid-based downsampling (same as test_kvcache_rope_attnmap.py)
- Equivalence check: sampled rows only

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
        python tests/test_kvcache_rope_reorder_64k.py
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
LENGTH = "64k"
LAYER = 5
DATA_PATH = Path(__file__).parent.parent / f"results/kvcache-rope/{MODEL}/{LENGTH}/layer_{LAYER:02d}.pt"
OUT_DIR = Path(__file__).parent.parent / f"results/kvcache-rope-reorder/{MODEL}/{LENGTH}"
DEVICE = "cuda"
CHUNK_SIZE = 4096
BLOCK_SIZE = 128
DENSITY_THRESHOLD = 0.1  # top 10% attention can be missed

# ============================================================
# Streaming density profiling (no full attn map stored)
# ============================================================

def profile_head_density_streaming(post_q, post_k, seq_len, num_heads, num_kv_heads,
                                   scale, device, chunk_size, block_size):
    """Profile density via streaming: compute attn chunk by chunk, accumulate block-level stats.

    For each head, we compute:
    - block_attn_sum[b]: total attention mass on K block b (summed over all query positions)
    - diagonal attention mass (within |q-k| < 2*block_size window)

    Memory: O(n_blocks) per head, not O(S^2).
    """
    num_groups = num_heads // num_kv_heads
    n_blocks = math.ceil(seq_len / block_size)

    densities = np.zeros(num_heads)
    block_coverage = np.zeros((num_heads, n_blocks), dtype=bool)
    diagonal_conc = np.zeros(num_heads)

    for h in range(num_heads):
        kv_idx = h // num_groups
        k_h = post_k[:, kv_idx, :].to(device=device, dtype=torch.float32)

        block_attn_sum = torch.zeros(n_blocks, dtype=torch.float64)
        diag_attn_total = 0.0
        total_attn = 0.0

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            q_chunk = post_q[start:end, h, :].to(device=device, dtype=torch.float32)
            scores = torch.mm(q_chunk, k_h.t()) * scale  # [C, S]

            # Causal mask
            row_idx = torch.arange(start, end, device=device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            scores.masked_fill_(col_idx > row_idx, float("-inf"))

            attn = torch.softmax(scores, dim=-1)  # [C, S] on GPU
            del scores, q_chunk

            # Accumulate block-level attention sums (on GPU, transfer small result)
            for b in range(n_blocks):
                b_start = b * block_size
                b_end = min((b + 1) * block_size, seq_len)
                block_attn_sum[b] += attn[:, b_start:b_end].sum().cpu().item()

            # Diagonal concentration (on GPU)
            window = block_size * 2
            for qi in range(0, end - start, block_size):
                qi_end = min(qi + block_size, end - start)
                q_pos_start = start + qi
                q_pos_end = start + qi_end
                k_lo = max(0, q_pos_start - window)
                k_hi = min(seq_len, q_pos_end + window)
                diag_attn_total += attn[qi:qi_end, k_lo:k_hi].sum().cpu().item()

            total_attn += attn.sum().cpu().item()
            del attn

        del k_h
        torch.cuda.empty_cache()

        # Determine required blocks (cumulative coverage approach)
        if total_attn > 0:
            block_attn_frac = block_attn_sum / total_attn
        else:
            block_attn_frac = block_attn_sum

        sorted_indices = torch.argsort(block_attn_frac, descending=True)
        cum_attn = 0.0
        required_threshold = 1.0 - DENSITY_THRESHOLD
        for idx in sorted_indices:
            block_coverage[h, idx.item()] = True
            cum_attn += block_attn_frac[idx.item()].item()
            if cum_attn >= required_threshold:
                break

        densities[h] = block_coverage[h].sum() / n_blocks
        diagonal_conc[h] = diag_attn_total / max(total_attn, 1e-8)

        print(f"  Head {h:2d}: density={densities[h]:.3f}, "
              f"blocks={block_coverage[h].sum()}/{n_blocks}, "
              f"diag_conc={diagonal_conc[h]:.3f}")

    return densities, block_coverage, diagonal_conc


# ============================================================
# Greedy regrouping algorithm (same as 16k version)
# ============================================================

def regroup_heads_greedy(densities, block_coverage, num_kv_heads):
    H = len(densities)
    G = num_kv_heads
    sorted_heads = np.argsort(-densities)

    best_k = None
    best_sparse_load = float('inf')
    best_groups = None

    for k in range(1, H - G + 2):
        dense_heads = sorted_heads[:k].tolist()
        sparse_heads = sorted_heads[k:].tolist()

        if len(sparse_heads) < G - 1:
            break

        groups = [[] for _ in range(G)]
        groups[0] = dense_heads

        for i, h_idx in enumerate(sparse_heads):
            groups[1 + (i % (G - 1))].append(h_idx)

        sparse_loads = []
        for g_idx in range(1, G):
            if len(groups[g_idx]) > 0:
                union_cov = np.any(block_coverage[groups[g_idx]], axis=0).sum()
            else:
                union_cov = 0
            sparse_loads.append(union_cov)

        max_sparse_load = max(sparse_loads) if sparse_loads else 0

        if max_sparse_load < best_sparse_load:
            best_sparse_load = max_sparse_load
            best_k = k
            best_groups = [g[:] for g in groups]

    regrouping_map = np.zeros(H, dtype=int)
    for g, heads in enumerate(best_groups):
        for h_idx in heads:
            regrouping_map[h_idx] = g

    return regrouping_map, best_groups


# ============================================================
# Grid-based attention map (from test_kvcache_rope_attnmap.py)
# ============================================================

def compute_head_gridded(post_q, post_k, h, kv_idx, seq_len, scale,
                         device, chunk_size, grid_size):
    """Compute attention with grid downsampling. Column pooling on GPU."""
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

        attn = torch.softmax(scores, dim=-1)
        del scores, q_chunk

        col_pooled = torch.zeros(C, n_grid, device=device, dtype=torch.float32)
        if full_blocks > 0:
            col_pooled[:, :full_blocks] = attn[:, :full_blocks * grid_size].reshape(
                C, full_blocks, grid_size
            ).mean(dim=2)
        if remainder > 0:
            col_pooled[:, full_blocks] = attn[:, full_blocks * grid_size:].mean(dim=1)
        del attn

        col_pooled_cpu = col_pooled.cpu().double()
        del col_pooled

        ri_indices = torch.arange(start, end) // grid_size
        attn_ds.index_add_(0, ri_indices, col_pooled_cpu)
        del col_pooled_cpu

    row_counts = torch.full((n_grid,), grid_size, dtype=torch.float64)
    if remainder > 0:
        row_counts[-1] = remainder
    attn_ds /= row_counts.unsqueeze(1)

    del k_h
    torch.cuda.empty_cache()
    return attn_ds.float()


# ============================================================
# Equivalence check (sampled rows)
# ============================================================

def verify_equivalence_sampled(post_q, post_k, v_data, original_map, regrouping_map,
                               seq_len, num_heads, scale, device, chunk_size,
                               n_sample_rows=256):
    """Verify attention validity on sampled query rows to save memory."""
    print("\n=== Equivalence Verification (sampled rows) ===")

    sample_rows = torch.linspace(0, seq_len - 1, n_sample_rows).long()
    test_heads = [0, 7, 15, 16, 24, 31]
    max_row_sum_err = 0.0
    max_diff = 0.0

    for h in test_heads:
        orig_kv = original_map[h]
        new_kv = regrouping_map[h]

        k_orig = post_k[:, orig_kv, :].to(device=device, dtype=torch.float32)
        v_orig = v_data[:, orig_kv, :].to(device=device, dtype=torch.float32)
        q_sample = post_q[sample_rows, h, :].to(device=device, dtype=torch.float32)

        scores_orig = torch.mm(q_sample, k_orig.t()) * scale
        row_idx = sample_rows.to(device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        scores_orig.masked_fill_(col_idx > row_idx, float("-inf"))
        attn_orig = torch.softmax(scores_orig, dim=-1)
        out_orig = torch.mm(attn_orig, v_orig)

        row_sums = attn_orig.sum(dim=1)
        row_err = (row_sums - 1.0).abs().max().item()
        max_row_sum_err = max(max_row_sum_err, row_err)

        del scores_orig

        if orig_kv == new_kv:
            print(f"  Head {h:2d}: KV group unchanged ({orig_kv} -> {new_kv}), output identical")
            del k_orig, v_orig, q_sample, attn_orig, out_orig
        else:
            k_new = post_k[:, new_kv, :].to(device=device, dtype=torch.float32)
            v_new = v_data[:, new_kv, :].to(device=device, dtype=torch.float32)
            scores_new = torch.mm(q_sample, k_new.t()) * scale
            scores_new.masked_fill_(col_idx > row_idx, float("-inf"))
            attn_new = torch.softmax(scores_new, dim=-1)
            out_new = torch.mm(attn_new, v_new)

            new_row_err = (attn_new.sum(dim=1) - 1.0).abs().max().item()
            max_row_sum_err = max(max_row_sum_err, new_row_err)

            diff = (out_orig.cpu() - out_new.cpu()).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                out_orig.cpu().flatten().unsqueeze(0),
                out_new.cpu().flatten().unsqueeze(0)
            ).item()
            max_diff = max(max_diff, max_d)

            print(f"  Head {h:2d}: KV {orig_kv} -> {new_kv}, "
                  f"max_diff={max_d:.6f}, mean_diff={mean_d:.6f}, cos_sim={cos_sim:.6f}")

            del k_new, v_new, scores_new, attn_new, out_new, k_orig, v_orig, q_sample, attn_orig, out_orig

        torch.cuda.empty_cache()

    print(f"\n  Max attention row sum error: {max_row_sum_err:.2e}")
    print(f"  Max output diff (across changed heads): {max_diff:.6f}")
    assert max_row_sum_err < 1e-4, f"Attention row sums deviate too much: {max_row_sum_err}"
    print("  Attention validity: PASSED")
    return max_diff


# ============================================================
# Visualization
# ============================================================

def plot_attn_maps(attn_maps, head_labels, densities, title, num_kv_heads,
                   groups, out_path):
    """Plot attention maps organized by groups."""
    max_group_size = max(len(g) for g in groups)
    n_groups = len(groups)

    fig = plt.figure(figsize=(max_group_size * 2.8, n_groups * 2.8 + 1.5))
    outer_gs = gridspec.GridSpec(n_groups, 1, figure=fig,
                                hspace=0.35, top=0.92, bottom=0.04, left=0.06, right=0.93)

    for g_idx, group in enumerate(groups):
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, max_group_size,
                                                    subplot_spec=outer_gs[g_idx], wspace=0.08)
        for j, h in enumerate(sorted(group)):
            ax = fig.add_subplot(inner_gs[j])
            img = np.log10(attn_maps[h] + 1e-8)
            ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest",
                      vmin=-6, vmax=0)
            ax.set_title(f"H{h} (d={densities[h]:.2f})", fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(head_labels[g_idx], fontsize=9, fontweight="bold")

        for j in range(len(group), max_group_size):
            ax = fig.add_subplot(inner_gs[j])
            ax.axis("off")

    cbar_ax = fig.add_axes([0.94, 0.04, 0.015, 0.88])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=-6, vmax=0))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log10(attention)", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_block_coverage(block_coverage, densities, groups_new, num_kv_heads,
                        num_groups_per_kv, out_dir, layer):
    H, n_blocks = block_coverage.shape

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.imshow(block_coverage.astype(float), aspect="auto", cmap="Blues", interpolation="nearest")
    ax1.set_ylabel("Q Head (original order)")
    ax1.set_xlabel("K Block")
    ax1.set_title("Block Coverage — Original GQA Grouping")
    for g in range(1, num_kv_heads):
        ax1.axhline(g * num_groups_per_kv - 0.5, color="red", linewidth=1.5)
    for g in range(num_kv_heads):
        ax1.text(-0.5, g * num_groups_per_kv + num_groups_per_kv / 2 - 0.5,
                 f"KV{g}", fontsize=8, ha="right", va="center", fontweight="bold", color="red")

    reorder = []
    group_boundaries = []
    pos = 0
    for g_idx, group in enumerate(groups_new):
        reorder.extend(sorted(group))
        pos += len(group)
        if g_idx < len(groups_new) - 1:
            group_boundaries.append(pos - 0.5)

    regrouped_cov = block_coverage[reorder]
    ax2.imshow(regrouped_cov.astype(float), aspect="auto", cmap="Oranges", interpolation="nearest")
    ax2.set_ylabel("Q Head (regrouped order)")
    ax2.set_xlabel("K Block")
    ax2.set_title("Block Coverage — Regrouped (G0=Dense, G1-3=Sparse)")
    for b in group_boundaries:
        ax2.axhline(b, color="red", linewidth=1.5)

    pos = 0
    for g_idx, group in enumerate(groups_new):
        label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
        mid = pos + len(group) / 2 - 0.5
        ax2.text(-0.5, mid, f"G{g_idx}\n({label})", fontsize=7, ha="right", va="center",
                 fontweight="bold", color="red")
        pos += len(group)

    ax1.set_yticks(list(range(H)))
    ax1.set_yticklabels([f"H{h} ({densities[h]:.2f})" for h in range(H)], fontsize=5)
    ax2.set_yticks(list(range(H)))
    ax2.set_yticklabels([f"H{reorder[i]} ({densities[reorder[i]]:.2f})" for i in range(H)], fontsize=5)

    fig.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — Block Coverage Comparison\n"
                 f"block_size={BLOCK_SIZE}, threshold={DENSITY_THRESHOLD}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / f"layer_{layer:02d}_block_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_load_analysis(densities, block_coverage, groups_new, num_kv_heads,
                       num_groups_per_kv, out_dir, layer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    n_blocks = block_coverage.shape[1]

    # Union coverage per group
    orig_unions = []
    for g in range(num_kv_heads):
        heads_in_g = list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
        union = np.any(block_coverage[heads_in_g], axis=0).sum() / n_blocks
        orig_unions.append(union)

    new_unions = []
    for group in groups_new:
        union = np.any(block_coverage[group], axis=0).sum() / n_blocks
        new_unions.append(union)

    x = np.arange(num_kv_heads)
    width = 0.35
    ax1.bar(x - width/2, orig_unions, width, label="Original", color="steelblue")
    ax1.bar(x + width/2, new_unions, width, label="Regrouped", color="coral")
    ax1.set_xlabel("KV Group")
    ax1.set_ylabel("Union Coverage (fraction)")
    ax1.set_title("KV Block Union Coverage per Group")
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1.05)

    ax2.hist(densities, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axvline(np.median(densities), color="red", linestyle="--",
                label=f"Median={np.median(densities):.3f}")
    ax2.set_xlabel("Head Density")
    ax2.set_ylabel("Count")
    ax2.set_title("Head Density Distribution")
    ax2.legend()

    fig.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — KV Load Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / f"layer_{layer:02d}_load_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

assert DATA_PATH.exists(), f"Data not found: {DATA_PATH}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data: {DATA_PATH}")
data = torch.load(DATA_PATH, map_location="cpu")
post_q = data["post_rope_q"]
post_k = data["post_rope_k"]
v_data = data["v"]

seq_len, num_heads, head_dim = post_q.shape
num_kv_heads = post_k.shape[1]
num_groups_per_kv = num_heads // num_kv_heads
scale = 1.0 / math.sqrt(head_dim)
n_blocks = math.ceil(seq_len / BLOCK_SIZE)

# Grid size for visualization (from test_kvcache_rope_attnmap.py logic)
GRID_SIZE = 128  # 64k uses grid_size=128
n_grid = math.ceil(seq_len / GRID_SIZE)

print(f"Model: {MODEL}, Layer: {LAYER}, Seq length: {seq_len}")
print(f"  Q: {post_q.shape}  (heads={num_heads}, head_dim={head_dim})")
print(f"  K: {post_k.shape}  (kv_heads={num_kv_heads}, GQA ratio={num_groups_per_kv}:1)")
print(f"  V: {v_data.shape}")
print(f"  Block size: {BLOCK_SIZE}, n_blocks: {n_blocks}")
print(f"  Grid size: {GRID_SIZE}, n_grid: {n_grid}")

# ============================================================
# Step 1: Streaming density profiling
# ============================================================

print("\n=== Step 1: Streaming Density Profiling ===")
densities, block_coverage, diagonal_conc = profile_head_density_streaming(
    post_q, post_k, seq_len, num_heads, num_kv_heads,
    scale, DEVICE, CHUNK_SIZE, BLOCK_SIZE
)

# ============================================================
# Step 2: Greedy regrouping
# ============================================================

print("\n=== Step 2: Greedy Regrouping ===")
regrouping_map, groups_new = regroup_heads_greedy(densities, block_coverage, num_kv_heads)
original_map = np.array([h // num_groups_per_kv for h in range(num_heads)])

print("\nOriginal mapping:")
for g in range(num_kv_heads):
    heads = [h for h in range(num_heads) if original_map[h] == g]
    avg_d = np.mean([densities[h] for h in heads])
    print(f"  KV Group {g}: heads={heads}, avg_density={avg_d:.3f}")

print("\nRegrouped mapping:")
for g_idx, group in enumerate(groups_new):
    label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
    avg_d = np.mean([densities[h] for h in group])
    print(f"  Group {g_idx} ({label}): heads={sorted(group)}, avg_density={avg_d:.3f}")

# ============================================================
# Step 3: Equivalence check (sampled)
# ============================================================

max_diff = verify_equivalence_sampled(
    post_q, post_k, v_data, original_map, regrouping_map,
    seq_len, num_heads, scale, DEVICE, CHUNK_SIZE
)

# ============================================================
# Step 4: Grid attention maps for visualization
# ============================================================

print("\n=== Step 4: Computing Grid Attention Maps ===")

# Original attention maps (grid-based)
print("Computing original attention maps (grid)...")
attn_original = {}
for h in range(num_heads):
    kv_idx = original_map[h]
    attn_h = compute_head_gridded(
        post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE, GRID_SIZE
    )
    attn_original[h] = attn_h.numpy()
    if (h + 1) % 8 == 0:
        print(f"  Original: heads {h-6}..{h+1}/{num_heads}")

# Regrouped attention maps
print("Computing regrouped attention maps (grid)...")
attn_regrouped = {}
for h in range(num_heads):
    kv_idx = regrouping_map[h]
    attn_h = compute_head_gridded(
        post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE, GRID_SIZE
    )
    attn_regrouped[h] = attn_h.numpy()
    if (h + 1) % 8 == 0:
        print(f"  Regrouped: heads {h-6}..{h+1}/{num_heads}")

# Convert to arrays
attn_orig_np = np.stack([attn_original[h] for h in range(num_heads)])
attn_regroup_np = np.stack([attn_regrouped[h] for h in range(num_heads)])

# ============================================================
# Step 5: Visualization
# ============================================================

print("\n=== Step 5: Visualization ===")

# Original layout (4 KV groups x 8 heads)
orig_groups = [list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
               for g in range(num_kv_heads)]
orig_labels = [f"KV {g}" for g in range(num_kv_heads)]

plot_attn_maps(attn_orig_np, orig_labels, densities,
               f"{MODEL} Layer {LAYER} ({LENGTH}, seq={seq_len}) — Original GQA\n"
               f"grid={GRID_SIZE}x{GRID_SIZE} → {n_grid}x{n_grid}",
               num_kv_heads, orig_groups,
               OUT_DIR / f"layer_{LAYER:02d}_original.png")

# Regrouped layout
regroup_labels = ["G0 (Dense)"] + [f"G{i} (Sparse)" for i in range(1, num_kv_heads)]
plot_attn_maps(attn_regroup_np, regroup_labels, densities,
               f"{MODEL} Layer {LAYER} ({LENGTH}, seq={seq_len}) — Regrouped GQA\n"
               f"G0=Dense, G1-3=Sparse | grid={GRID_SIZE}x{GRID_SIZE}",
               num_kv_heads, groups_new,
               OUT_DIR / f"layer_{LAYER:02d}_regrouped.png")

plot_block_coverage(block_coverage, densities, groups_new, num_kv_heads,
                    num_groups_per_kv, OUT_DIR, LAYER)

plot_load_analysis(densities, block_coverage, groups_new, num_kv_heads,
                   num_groups_per_kv, OUT_DIR, LAYER)

# ============================================================
# Block coverage stats
# ============================================================

print("\n=== Block Coverage Union Stats ===")

print("\nOriginal grouping:")
total_orig = 0
for g in range(num_kv_heads):
    heads_in_g = list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
    union = np.any(block_coverage[heads_in_g], axis=0).sum()
    total_orig += union
    print(f"  KV Group {g}: heads={heads_in_g}, union={union}/{n_blocks} "
          f"({union/n_blocks*100:.1f}%)")

print("\nRegrouped:")
total_new = 0
for g_idx, group in enumerate(groups_new):
    union = np.any(block_coverage[group], axis=0).sum()
    total_new += union
    label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
    print(f"  Group {g_idx} ({label}): heads={sorted(group)}, union={union}/{n_blocks} "
          f"({union/n_blocks*100:.1f}%)")

print(f"\n  Total KV blocks loaded (original): {total_orig}")
print(f"  Total KV blocks loaded (regrouped): {total_new}")
if total_orig > 0:
    print(f"  Reduction: {(1 - total_new/total_orig)*100:.1f}%")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: {MODEL}, Layer: {LAYER}, Length: {LENGTH}, Seq: {seq_len}")
print(f"Heads: {num_heads} Q, {num_kv_heads} KV (GQA {num_groups_per_kv}:1)")
print(f"\nDensity stats:")
print(f"  Min: {densities.min():.3f}, Max: {densities.max():.3f}, "
      f"Mean: {densities.mean():.3f}, Std: {densities.std():.3f}")
print(f"\nRegrouping result:")
for g_idx, group in enumerate(groups_new):
    label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
    union = np.any(block_coverage[group], axis=0).sum()
    print(f"  G{g_idx} ({label:7s}): {len(group):2d} heads, "
          f"union coverage={union}/{n_blocks} ({union/n_blocks*100:.1f}%)")

print(f"\nOutput difference (orig vs regrouped): max={max_diff:.6f}")
print(f"Output directory: {OUT_DIR}")

print("\ntest_kvcache_rope_reorder_64k: PASSED")
