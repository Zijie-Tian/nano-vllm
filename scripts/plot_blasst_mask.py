import re
import torch
import matplotlib.pyplot as plt
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures


def get_num(s):
    if "current" in s:
        return 999999  # Ensure 'current' always sorts to the very end
    match = re.search(r"\d+", s)
    return int(match.group()) if match else -1


def process_single_layer(args):
    layer_id, q_chunk_dirs, output_path = args
    q_strips = []

    for q_dir in q_chunk_dirs:
        layer_file = q_dir / f"layer_{layer_id}.pt"
        if not layer_file.exists():
            continue

        try:
            data = torch.load(layer_file, map_location="cpu")
        except Exception as e:
            print(f"Error loading {layer_file}: {e}")
            continue

        # Sort kvchunks by index
        kv_keys = sorted(data.keys(), key=lambda x: get_num(x))
        # Each tensor is [Q_blocks, Heads, KV_subblocks]
        kv_strip = torch.cat([data[k] for k in kv_keys], dim=2)
        q_strips.append(kv_strip)

    if not q_strips:
        return layer_id

    # Find maximum KV width to pad
    max_kv_width = max(strip.shape[2] for strip in q_strips)

    padded_strips = []
    for strip in q_strips:
        if strip.shape[2] < max_kv_width:
            padding = torch.zeros(
                (strip.shape[0], strip.shape[1], max_kv_width - strip.shape[2]),
                dtype=strip.dtype,
            )
            strip = torch.cat([strip, padding], dim=2)
        padded_strips.append(strip)

    # Full layer mask: [Total_Q_blocks, Heads, Total_KV_subblocks]
    layer_mask = torch.cat(padded_strips, dim=0)
    num_heads = layer_mask.shape[1]

    # Plot all heads in one figure
    cols = 8
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True
    )
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        head_mask = layer_mask[:, h, :].float().numpy()

        ax.imshow(
            head_mask, cmap="viridis", aspect="auto", interpolation="nearest"
        )
        ax.set_title(f"Head {h}", fontsize=10)
        ax.axis("off")

    # Hide remaining empty subplots
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(
        f"BLASST Mask - Layer {layer_id} (All Heads)\nRows: Query Blocks, Cols: KV Subblocks",
        fontsize=16,
    )

    save_path = output_path / f"layer_{layer_id}.jpg"
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

    return layer_id


def plot_masks(source_dir, output_dir, workers):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Clean up old output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    # 1. Get sorted list of query chunks
    q_chunk_dirs = sorted(
        [
            d
            for d in source_path.iterdir()
            if d.is_dir() and d.name.startswith("q_chunk_")
        ],
        key=lambda x: get_num(x.name),
    )

    if not q_chunk_dirs:
        print("No q_chunk directories found.")
        return

    # 2. Find all unique layers
    layer_ids = set()
    for q_dir in q_chunk_dirs:
        for f in q_dir.glob("layer_*.pt"):
            layer_ids.add(get_num(f.name))
    layer_ids = sorted(list(layer_ids))

    print(f"Found {len(q_chunk_dirs)} query chunks and {len(layer_ids)} layers.")
    print(f"Using {workers} processes for rendering...")

    # 3. Parallel processing
    tasks = [(layer_id, q_chunk_dirs, output_path) for layer_id in layer_ids]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(process_single_layer, tasks),
                total=len(tasks),
                desc="Rendering Maps",
            )
        )

    print(f"Done! Combined maps saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="results/chunked_mask", help="Source directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/chunked_mask_map",
        help="Output directory",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )
    args = parser.parse_args()

    plot_masks(args.source, args.output, args.workers)
