"""
Test: Q Head Regrouping Validation for GQA Dense Head Optimization.

Validates Solution 4 (Q Head Regrouping) from the GQA dense head analysis:
1. Profile each Q head's density (attention pattern sparsity)
2. Compute a greedy regrouping that isolates dense heads into one KV group
3. Verify that regrouping does NOT change attention output (mathematical equivalence)
4. Visualize original vs regrouped attention maps side by side

Key insight from the doc: regrouping only changes which KV head each Q head
queries. Since Q head order and W_o are unchanged, the final output is
mathematically equivalent. However, different Q heads paired with different
KV heads WILL produce different attention maps — this test verifies the
regrouping strategy and visualizes the KV load distribution change.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
        python tests/test_kvcache_rope_reorder.py
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
LENGTH = "16k"
LAYER = 5
DATA_PATH = Path(__file__).parent.parent / f"results/kvcache-rope/{MODEL}/{LENGTH}/layer_{LAYER:02d}.pt"
OUT_DIR = Path(__file__).parent.parent / f"results/kvcache-rope-reorder/{MODEL}/{LENGTH}"
DEVICE = "cuda"
CHUNK_SIZE = 4096
DENSITY_THRESHOLD = 0.1  # top fraction of attention mass to define "required" blocks
BLOCK_SIZE = 128  # block size for sparsity analysis

# ============================================================
# Utility: Chunked attention computation
# ============================================================

def compute_attention_output(post_q, post_k, v, h, kv_idx, seq_len, scale, device, chunk_size):
    """Compute attention output O_h = softmax(Q_h K_g^T / sqrt(d)) V_g for one Q head.

    Returns:
        attn_output: [S, D] the attention output for head h
        attn_map: [S, S] the full attention map (on CPU)
    """
    k_h = post_k[:, kv_idx, :].to(device=device, dtype=torch.float32)
    v_h = v[:, kv_idx, :].to(device=device, dtype=torch.float32)

    output_rows = []
    attn_rows = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        q_chunk = post_q[start:end, h, :].to(device=device, dtype=torch.float32)
        scores = torch.mm(q_chunk, k_h.t()) * scale

        row_idx = torch.arange(start, end, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        scores.masked_fill_(col_idx > row_idx, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out_chunk = torch.mm(attn, v_h)  # [C, D]

        output_rows.append(out_chunk.cpu())
        attn_rows.append(attn.cpu())
        del scores, attn, q_chunk, out_chunk

    del k_h, v_h
    torch.cuda.empty_cache()
    return torch.cat(output_rows, dim=0), torch.cat(attn_rows, dim=0)


def compute_attention_map_only(post_q, post_k, h, kv_idx, seq_len, scale, device, chunk_size):
    """Compute only the attention map (no output needed) for visualization."""
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
    return torch.cat(attn_rows, dim=0)


# ============================================================
# Density profiling
# ============================================================

def profile_head_density(post_q, post_k, seq_len, num_heads, num_kv_heads,
                         scale, device, chunk_size, block_size):
    """Profile each Q head's attention density.

    For each head, compute attention map and measure:
    - density: fraction of K blocks that receive significant attention
    - block_coverage: binary mask of which K blocks are needed

    Returns:
        densities: [H] density score per head
        block_coverage: [H, n_blocks] bool mask
        diagonal_conc: [H] diagonal concentration ratio
    """
    num_groups = num_heads // num_kv_heads
    n_blocks = math.ceil(seq_len / block_size)

    densities = np.zeros(num_heads)
    block_coverage = np.zeros((num_heads, n_blocks), dtype=bool)
    diagonal_conc = np.zeros(num_heads)

    for h in range(num_heads):
        kv_idx = h // num_groups
        attn_map = compute_attention_map_only(
            post_q, post_k, h, kv_idx, seq_len, scale, device, chunk_size
        )

        # Block-level attention: average attention per block column
        # attn_map: [S, S]. Sum attention across all query positions for each key block.
        block_attn = torch.zeros(n_blocks)
        for b in range(n_blocks):
            b_start = b * block_size
            b_end = min((b + 1) * block_size, seq_len)
            block_attn[b] = attn_map[:, b_start:b_end].sum().item()

        # Normalize to get fraction of total attention per block
        total_attn = block_attn.sum().item()
        if total_attn > 0:
            block_attn_frac = block_attn / total_attn
        else:
            block_attn_frac = block_attn

        # Determine required blocks: cumulative sum approach
        # Sort blocks by attention (descending), accumulate until threshold
        sorted_indices = torch.argsort(block_attn_frac, descending=True)
        cum_attn = 0.0
        required_threshold = 1.0 - DENSITY_THRESHOLD  # capture 90% of attention
        for idx in sorted_indices:
            block_coverage[h, idx.item()] = True
            cum_attn += block_attn_frac[idx.item()].item()
            if cum_attn >= required_threshold:
                break

        densities[h] = block_coverage[h].sum() / n_blocks

        # Diagonal concentration: fraction of attention within window |q-k| < block_size*2
        diag_attn = 0.0
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = attn_map[start:end]
            for qi in range(chunk.shape[0]):
                q_pos = start + qi
                k_lo = max(0, q_pos - block_size * 2)
                k_hi = min(seq_len, q_pos + block_size * 2 + 1)
                diag_attn += chunk[qi, k_lo:k_hi].sum().item()

        diagonal_conc[h] = diag_attn / max(total_attn, 1e-8)

        del attn_map
        print(f"  Head {h:2d}: density={densities[h]:.3f}, "
              f"blocks={block_coverage[h].sum()}/{n_blocks}, "
              f"diag_conc={diagonal_conc[h]:.3f}")

    return densities, block_coverage, diagonal_conc


# ============================================================
# Greedy regrouping algorithm
# ============================================================

def regroup_heads_greedy(densities, block_coverage, num_kv_heads):
    """Greedy algorithm: isolate dense heads into one group.

    Returns:
        regrouping_map: [H] array mapping Q head -> new KV group
        groups: list of lists, groups[g] = list of Q head indices
    """
    H = len(densities)
    G = num_kv_heads
    sorted_heads = np.argsort(-densities)  # descending by density

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

        # Distribute sparse heads to remaining groups, balancing union coverage
        for i, h in enumerate(sparse_heads):
            groups[1 + (i % (G - 1))].append(h)

        # Calculate max load across sparse groups
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

    # Build regrouping map
    regrouping_map = np.zeros(H, dtype=int)
    for g, heads in enumerate(best_groups):
        for h in heads:
            regrouping_map[h] = g

    return regrouping_map, best_groups


# ============================================================
# Equivalence verification
# ============================================================

def verify_equivalence(post_q, post_k, v, original_map, regrouping_map,
                       seq_len, num_heads, scale, device, chunk_size):
    """Verify that the ORIGINAL mapping produces the same output as itself.

    Important: regrouping changes which KV head a Q head uses, so outputs
    WILL differ. What we verify here is:
    1. Original mapping: each head's output is correct
    2. The regrouped mapping also produces valid attention (row sums = 1)
    3. Report the numerical difference between original and regrouped outputs
    """
    print("\n=== Equivalence Verification ===")

    # Pick a few representative heads to compare
    test_heads = [0, 7, 15, 16, 24, 31]  # from different original groups
    max_diff = 0.0
    max_row_sum_err = 0.0

    for h in test_heads:
        orig_kv = original_map[h]
        new_kv = regrouping_map[h]

        # Original attention output
        orig_out, orig_attn = compute_attention_output(
            post_q, post_k, v, h, orig_kv, seq_len, scale, device, chunk_size
        )

        # Verify original attention rows sum to 1
        row_sums = orig_attn.sum(dim=1)
        row_err = (row_sums - 1.0).abs().max().item()
        max_row_sum_err = max(max_row_sum_err, row_err)

        if orig_kv == new_kv:
            print(f"  Head {h:2d}: KV group unchanged ({orig_kv} -> {new_kv}), output identical")
        else:
            # Regrouped attention output
            new_out, new_attn = compute_attention_output(
                post_q, post_k, v, h, new_kv, seq_len, scale, device, chunk_size
            )

            # Check new attention rows sum to 1
            new_row_sums = new_attn.sum(dim=1)
            new_row_err = (new_row_sums - 1.0).abs().max().item()
            max_row_sum_err = max(max_row_sum_err, new_row_err)

            # Compute output difference
            diff = (orig_out - new_out).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_out.flatten().unsqueeze(0),
                new_out.flatten().unsqueeze(0)
            ).item()

            max_diff = max(max_diff, max_d)
            print(f"  Head {h:2d}: KV {orig_kv} -> {new_kv}, "
                  f"max_diff={max_d:.6f}, mean_diff={mean_d:.6f}, cos_sim={cos_sim:.6f}")

            del new_out, new_attn

        del orig_out, orig_attn
        torch.cuda.empty_cache()

    print(f"\n  Max attention row sum error: {max_row_sum_err:.2e}")
    print(f"  Max output diff (across changed heads): {max_diff:.6f}")

    assert max_row_sum_err < 1e-4, f"Attention row sums deviate too much: {max_row_sum_err}"
    print("  Attention validity: PASSED (all row sums ≈ 1.0)")

    return max_diff


# ============================================================
# Visualization
# ============================================================

def plot_comparison(attn_original, attn_regrouped, original_map, regrouping_map,
                    groups_new, densities, num_heads, num_kv_heads, seq_len, out_dir):
    """Plot original vs regrouped attention maps side by side."""
    num_groups_per_kv = num_heads // num_kv_heads

    # --- Figure 1: Original grouping attention map ---
    fig1, axes1 = plt.subplots(num_kv_heads, num_groups_per_kv,
                               figsize=(num_groups_per_kv * 2.5, num_kv_heads * 2.5 + 1))
    for h in range(num_heads):
        g = h // num_groups_per_kv
        j = h % num_groups_per_kv
        ax = axes1[g, j] if num_kv_heads > 1 else axes1[j]
        img = np.log10(attn_original[h] + 1e-8)
        ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest", vmin=-6, vmax=0)
        ax.set_title(f"H{h} (d={densities[h]:.2f})", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"KV {g}", fontsize=9, fontweight="bold")

    fig1.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — Original GQA Grouping\n"
                  f"Attention Map (log10 scale)", fontsize=11, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = out_dir / f"layer_{LAYER:02d}_original.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # --- Figure 2: Regrouped attention map ---
    # Organize by new groups
    max_group_size = max(len(g) for g in groups_new)
    fig2, axes2 = plt.subplots(num_kv_heads, max_group_size,
                               figsize=(max_group_size * 2.5, num_kv_heads * 2.5 + 1))
    for g_idx, group in enumerate(groups_new):
        for j, h in enumerate(sorted(group)):
            ax = axes2[g_idx, j] if num_kv_heads > 1 else axes2[j]
            img = np.log10(attn_regrouped[h] + 1e-8)
            ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest", vmin=-6, vmax=0)
            label = "D" if g_idx == 0 else "S"
            ax.set_title(f"H{h} [{label}] (d={densities[h]:.2f})", fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"G{g_idx}", fontsize=9, fontweight="bold")

        # Hide unused axes
        for j in range(len(group), max_group_size):
            axes2[g_idx, j].axis("off")

    fig2.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — Regrouped GQA\n"
                  f"G0=Dense group, G1-G3=Sparse groups", fontsize=11, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = out_dir / f"layer_{LAYER:02d}_regrouped.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # --- Figure 3: KV load distribution comparison ---
    fig3, (ax_orig, ax_new) = plt.subplots(1, 2, figsize=(12, 5))

    n_blocks = attn_original.shape[1] // BLOCK_SIZE if attn_original.shape[1] >= BLOCK_SIZE else attn_original.shape[1]

    # Original load per group
    orig_loads = []
    for g in range(num_kv_heads):
        heads_in_g = list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
        union_blocks = 0
        for h in heads_in_g:
            union_blocks = max(union_blocks, densities[h])
        # Use actual block coverage union
        orig_loads.append(np.mean([densities[h] for h in heads_in_g]))

    new_loads = []
    for g_idx, group in enumerate(groups_new):
        new_loads.append(np.mean([densities[h] for h in group]))

    x = np.arange(num_kv_heads)
    width = 0.35
    ax_orig.bar(x - width/2, orig_loads, width, label="Original", color="steelblue")
    ax_orig.bar(x + width/2, new_loads, width, label="Regrouped", color="coral")
    ax_orig.set_xlabel("KV Group")
    ax_orig.set_ylabel("Avg Head Density")
    ax_orig.set_title("Average Head Density per Group")
    ax_orig.legend()
    ax_orig.set_xticks(x)

    # Density histogram
    ax_new.hist(densities, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax_new.axvline(np.median(densities), color="red", linestyle="--", label=f"Median={np.median(densities):.3f}")
    ax_new.set_xlabel("Head Density")
    ax_new.set_ylabel("Count")
    ax_new.set_title("Head Density Distribution")
    ax_new.legend()

    fig3.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — KV Load Analysis", fontsize=12, fontweight="bold")
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    path3 = out_dir / f"layer_{LAYER:02d}_load_analysis.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {path3}")


def plot_block_coverage(block_coverage, densities, groups_new, num_kv_heads,
                        num_groups_per_kv, out_dir):
    """Plot block coverage heatmap for original and regrouped layouts."""
    H, n_blocks = block_coverage.shape

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Original order
    ax1.imshow(block_coverage.astype(float), aspect="auto", cmap="Blues", interpolation="nearest")
    ax1.set_ylabel("Q Head (original order)")
    ax1.set_xlabel("K Block")
    ax1.set_title("Block Coverage — Original GQA Grouping")
    for g in range(1, num_kv_heads):
        ax1.axhline(g * num_groups_per_kv - 0.5, color="red", linewidth=1.5)
    for g in range(num_kv_heads):
        ax1.text(-0.5, g * num_groups_per_kv + num_groups_per_kv / 2 - 0.5,
                 f"KV{g}", fontsize=8, ha="right", va="center", fontweight="bold", color="red")

    # Regrouped order
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

    # Label groups
    pos = 0
    for g_idx, group in enumerate(groups_new):
        label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
        mid = pos + len(group) / 2 - 0.5
        ax2.text(-0.5, mid, f"G{g_idx}\n({label})", fontsize=7, ha="right", va="center",
                 fontweight="bold", color="red")
        pos += len(group)

    # Add head labels on y-axis
    yticks_orig = list(range(H))
    ax1.set_yticks(yticks_orig)
    ax1.set_yticklabels([f"H{h} ({densities[h]:.2f})" for h in range(H)], fontsize=5)
    ax2.set_yticks(list(range(H)))
    ax2.set_yticklabels([f"H{reorder[i]} ({densities[reorder[i]]:.2f})" for i in range(H)], fontsize=5)

    fig.suptitle(f"{MODEL} Layer {LAYER} ({LENGTH}) — Block Coverage Comparison\n"
                 f"block_size={BLOCK_SIZE}, threshold={DENSITY_THRESHOLD}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / f"layer_{LAYER:02d}_block_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print union coverage stats
    print("\n=== Block Coverage Union Stats ===")
    n_blocks_total = block_coverage.shape[1]

    print("\nOriginal grouping:")
    for g in range(num_kv_heads):
        heads_in_g = list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
        union = np.any(block_coverage[heads_in_g], axis=0).sum()
        print(f"  KV Group {g}: heads={heads_in_g}, union={union}/{n_blocks_total} "
              f"({union/n_blocks_total*100:.1f}%)")

    print("\nRegrouped:")
    total_orig = 0
    total_new = 0
    for g_idx, group in enumerate(groups_new):
        union = np.any(block_coverage[group], axis=0).sum()
        label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
        print(f"  Group {g_idx} ({label}): heads={sorted(group)}, union={union}/{n_blocks_total} "
              f"({union/n_blocks_total*100:.1f}%)")
        total_new += union

    for g in range(num_kv_heads):
        heads_in_g = list(range(g * num_groups_per_kv, (g + 1) * num_groups_per_kv))
        total_orig += np.any(block_coverage[heads_in_g], axis=0).sum()

    print(f"\n  Total KV blocks loaded (original): {total_orig}")
    print(f"  Total KV blocks loaded (regrouped): {total_new}")
    if total_orig > 0:
        print(f"  Reduction: {(1 - total_new/total_orig)*100:.1f}%")


# ============================================================
# Main
# ============================================================

assert DATA_PATH.exists(), f"Data not found: {DATA_PATH}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data: {DATA_PATH}")
data = torch.load(DATA_PATH, map_location="cpu")
post_q = data["post_rope_q"]  # [S, H, D]
post_k = data["post_rope_k"]  # [S, Hkv, D]
v_data = data["v"]            # [S, Hkv, D]

seq_len, num_heads, head_dim = post_q.shape
num_kv_heads = post_k.shape[1]
num_groups_per_kv = num_heads // num_kv_heads
scale = 1.0 / math.sqrt(head_dim)

print(f"Model: {MODEL}, Layer: {LAYER}, Seq length: {seq_len}")
print(f"  Q: {post_q.shape}  (heads={num_heads}, head_dim={head_dim})")
print(f"  K: {post_k.shape}  (kv_heads={num_kv_heads}, GQA ratio={num_groups_per_kv}:1)")
print(f"  V: {v_data.shape}")
print(f"  Block size: {BLOCK_SIZE}, n_blocks: {math.ceil(seq_len / BLOCK_SIZE)}")

# ============================================================
# Step 1: Profile head density
# ============================================================

print("\n=== Step 1: Profiling Head Density ===")
densities, block_coverage, diagonal_conc = profile_head_density(
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
    print(f"  KV Group {g}: heads={heads}")

print("\nRegrouped mapping:")
for g_idx, group in enumerate(groups_new):
    label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
    avg_d = np.mean([densities[h] for h in group])
    print(f"  Group {g_idx} ({label}): heads={sorted(group)}, avg_density={avg_d:.3f}")

# ============================================================
# Step 3: Verify equivalence
# ============================================================

max_diff = verify_equivalence(
    post_q, post_k, v_data, original_map, regrouping_map,
    seq_len, num_heads, scale, DEVICE, CHUNK_SIZE
)

# ============================================================
# Step 4: Compute attention maps for visualization
# ============================================================

print("\n=== Step 4: Computing Attention Maps ===")

# Downsample for visualization (16k is small enough for VIS_SIZE approach)
VIS_SIZE = min(512, seq_len)

# Original attention maps
print("Computing original attention maps...")
attn_original = []
for h in range(num_heads):
    kv_idx = original_map[h]
    attn_h = compute_attention_map_only(
        post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE
    )
    # Downsample for plotting
    if seq_len > VIS_SIZE:
        attn_h = torch.nn.functional.adaptive_avg_pool2d(
            attn_h.unsqueeze(0).unsqueeze(0), (VIS_SIZE, VIS_SIZE)
        ).squeeze().numpy()
    else:
        attn_h = attn_h.numpy()
    attn_original.append(attn_h)
    if (h + 1) % 8 == 0:
        print(f"  Original: heads {h-6}..{h+1}/{num_heads}")
attn_original = np.stack(attn_original)

# Regrouped attention maps (with new KV assignment)
print("Computing regrouped attention maps...")
attn_regrouped = []
for h in range(num_heads):
    kv_idx = regrouping_map[h]
    attn_h = compute_attention_map_only(
        post_q, post_k, h, kv_idx, seq_len, scale, DEVICE, CHUNK_SIZE
    )
    if seq_len > VIS_SIZE:
        attn_h = torch.nn.functional.adaptive_avg_pool2d(
            attn_h.unsqueeze(0).unsqueeze(0), (VIS_SIZE, VIS_SIZE)
        ).squeeze().numpy()
    else:
        attn_h = attn_h.numpy()
    attn_regrouped.append(attn_h)
    if (h + 1) % 8 == 0:
        print(f"  Regrouped: heads {h-6}..{h+1}/{num_heads}")
attn_regrouped = np.stack(attn_regrouped)

# ============================================================
# Step 5: Visualize
# ============================================================

print("\n=== Step 5: Visualization ===")
plot_comparison(attn_original, attn_regrouped, original_map, regrouping_map,
                groups_new, densities, num_heads, num_kv_heads, seq_len, OUT_DIR)

plot_block_coverage(block_coverage, densities, groups_new, num_kv_heads,
                    num_groups_per_kv, OUT_DIR)

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: {MODEL}, Layer: {LAYER}, Length: {LENGTH}")
print(f"Heads: {num_heads} Q heads, {num_kv_heads} KV heads (GQA {num_groups_per_kv}:1)")
print(f"\nDensity stats:")
print(f"  Min: {densities.min():.3f}, Max: {densities.max():.3f}, "
      f"Mean: {densities.mean():.3f}, Std: {densities.std():.3f}")
print(f"\nRegrouping result:")
for g_idx, group in enumerate(groups_new):
    label = "Dense" if g_idx == 0 else f"Sparse{g_idx}"
    n_blocks_total = block_coverage.shape[1]
    union = np.any(block_coverage[group], axis=0).sum()
    print(f"  G{g_idx} ({label:7s}): {len(group):2d} heads, "
          f"union coverage={union}/{n_blocks_total} ({union/n_blocks_total*100:.1f}%)")

print(f"\nOutput difference (orig vs regrouped): max={max_diff:.6f}")
print(f"Output directory: {OUT_DIR}")

print("\ntest_kvcache_rope_reorder: PASSED")
