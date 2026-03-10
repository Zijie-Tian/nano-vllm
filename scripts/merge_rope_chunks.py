#!/usr/bin/env python3
"""
Merge per-chunk RoPE data into per-layer files.

Usage:
    python scripts/merge_rope_chunks.py <chunk_dir> <output_dir> [--delete-chunks]

--delete-chunks: delete chunk files for each layer after merging (saves disk space)
"""

import os
import sys
import glob
import torch


def merge_chunks(chunk_dir: str, output_dir: str, delete_chunks: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # Find all chunk files: layer_NN_chunk_MMMM.pt
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "layer_*_chunk_*.pt")))
    if not chunk_files:
        print(f"[MERGE] No chunk files found in {chunk_dir}")
        return

    # Group by layer
    layers = {}
    for f in chunk_files:
        basename = os.path.basename(f)
        # layer_00_chunk_0000.pt
        parts = basename.replace(".pt", "").split("_")
        layer_id = int(parts[1])
        chunk_idx = int(parts[3])
        if layer_id not in layers:
            layers[layer_id] = []
        layers[layer_id].append((chunk_idx, f))

    print(
        f"[MERGE] Found {len(chunk_files)} chunk files across {len(layers)} layers"
        + (" (will delete chunks after merge)" if delete_chunks else "")
    )

    # Merge each layer
    for layer_id in sorted(layers.keys()):
        chunks = sorted(layers[layer_id], key=lambda x: x[0])

        all_data = {
            "pre_rope_q": [],
            "pre_rope_k": [],
            "post_rope_q": [],
            "post_rope_k": [],
            "v": [],
            "positions": [],
        }

        for chunk_idx, chunk_file in chunks:
            data = torch.load(chunk_file, map_location="cpu", weights_only=True)
            for key in all_data:
                all_data[key].append(data[key])

        # Concatenate along sequence dimension (dim=0)
        merged = {}
        for key in all_data:
            merged[key] = torch.cat(all_data[key], dim=0)

        save_path = os.path.join(output_dir, f"layer_{layer_id:02d}.pt")
        torch.save(merged, save_path)

        total_tokens = merged["pre_rope_q"].shape[0]
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(
            f"[MERGE] Layer {layer_id:2d}: {len(chunks)} chunks, "
            f"{total_tokens} tokens, {file_size_mb:.1f} MB -> {save_path}"
        )

        # Delete chunk files for this layer to free disk space
        if delete_chunks:
            for _, chunk_file in chunks:
                os.remove(chunk_file)

    print(f"\n[MERGE] Done: {len(layers)} layers merged to {output_dir}")


if __name__ == "__main__":
    delete_chunks = "--delete-chunks" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--delete-chunks"]
    if len(args) != 2:
        print(f"Usage: {sys.argv[0]} <chunk_dir> <output_dir> [--delete-chunks]")
        sys.exit(1)

    merge_chunks(args[0], args[1], delete_chunks=delete_chunks)
