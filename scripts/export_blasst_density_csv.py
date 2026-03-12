#!/usr/bin/env python3
"""
Export BLASST density logs to per-layer CSV files.

Parses BLASST log output and generates per-layer CSV files for both
compute density and KV cache density, with rows=chunks and columns=heads.

Usage:
    python scripts/export_blasst_density_csv.py <log_file> <output_dir>

Example:
    python scripts/export_blasst_density_csv.py /tmp/blasst.log results/density/dynamic_lambda

Output files:
    <output_dir>/compute_layer_XX.csv   - per-head compute density (%)
    <output_dir>/kvcache_layer_XX.csv   - per-head KV cache density (%)
    <output_dir>/summary.csv            - scalar summary (layer, chunk, compute%, kv%)
"""

import re
import os
import sys
import csv
from collections import defaultdict


def parse_head_values(head_str):
    """Parse 'H0=xx.x%, H1=yy.y%, ...' into list of floats."""
    vals = re.findall(r"H\d+=([\d.]+)%", head_str)
    return [float(v) for v in vals]


def parse_log(log_file):
    """Parse a BLASST density log file.

    Returns:
        stats: dict[layer][chunk] = (compute_density, kv_density)
        kv_heads: dict[layer][chunk] = [h0, h1, ..., h31]
        comp_heads: dict[layer][chunk] = [h0, h1, ..., h31]
    """
    stats_pattern = re.compile(
        r"\[BLASST\] Layer (\d+) Chunk (\d+) Stats: "
        r"Compute Density=([\d.]+)%, "
        r"Required KV Density\(avg\)=([\d.]+)%"
    )
    kv_head_pattern = re.compile(
        r"\[BLASST\] Layer (\d+) Chunk (\d+) Per-Head KV Density: (.+)"
    )
    comp_head_pattern = re.compile(
        r"\[BLASST\] Layer (\d+) Chunk (\d+) Per-Head Compute Density: (.+)"
    )

    stats = defaultdict(dict)
    kv_heads = defaultdict(dict)
    comp_heads = defaultdict(dict)

    with open(log_file) as f:
        for line in f:
            m = stats_pattern.search(line)
            if m:
                layer, chunk = int(m.group(1)), int(m.group(2))
                cd, kv = float(m.group(3)), float(m.group(4))
                stats[layer][chunk] = (cd, kv)
                continue
            m = kv_head_pattern.search(line)
            if m:
                layer, chunk = int(m.group(1)), int(m.group(2))
                kv_heads[layer][chunk] = parse_head_values(m.group(3))
                continue
            m = comp_head_pattern.search(line)
            if m:
                layer, chunk = int(m.group(1)), int(m.group(2))
                comp_heads[layer][chunk] = parse_head_values(m.group(3))
                continue

    return stats, kv_heads, comp_heads


def export_csvs(stats, kv_heads, comp_heads, output_dir):
    """Export per-layer CSV files and summary."""
    os.makedirs(output_dir, exist_ok=True)

    num_layers = max(stats.keys()) + 1
    chunks = sorted(set(c for layer_data in stats.values() for c in layer_data.keys()))
    num_heads = 0
    if kv_heads:
        first_layer = next(iter(kv_heads.values()))
        first_chunk = next(iter(first_layer.values()))
        num_heads = len(first_chunk)

    print(f"Parsed: {num_layers} layers, {len(chunks)} chunks, {num_heads} heads")
    print(f"Has per-head compute density: {len(comp_heads) > 0}")
    print(f"Output dir: {output_dir}")

    # Export per-layer CSVs
    for layer in range(num_layers):
        # KV density CSV
        kv_file = os.path.join(output_dir, f"kvcache_layer_{layer:02d}.csv")
        with open(kv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["chunk"] + [f"H{i}" for i in range(num_heads)] + ["avg"]
            writer.writerow(header)
            for c in chunks:
                if c in kv_heads.get(layer, {}):
                    vals = kv_heads[layer][c]
                    avg = sum(vals) / len(vals)
                    writer.writerow(
                        [f"C{c}"] + [f"{v:.2f}" for v in vals] + [f"{avg:.2f}"]
                    )
                else:
                    writer.writerow([f"C{c}"] + ["N/A"] * num_heads + ["N/A"])

        # Compute density CSV
        comp_file = os.path.join(output_dir, f"compute_layer_{layer:02d}.csv")
        with open(comp_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["chunk"] + [f"H{i}" for i in range(num_heads)] + ["avg"]
            writer.writerow(header)
            for c in chunks:
                if c in comp_heads.get(layer, {}):
                    vals = comp_heads[layer][c]
                    avg = sum(vals) / len(vals)
                    writer.writerow(
                        [f"C{c}"] + [f"{v:.2f}" for v in vals] + [f"{avg:.2f}"]
                    )
                elif c in stats.get(layer, {}):
                    # Fallback: use scalar compute density for all heads
                    cd = stats[layer][c][0]
                    writer.writerow(
                        [f"C{c}"] + [f"{cd:.2f}"] * num_heads + [f"{cd:.2f}"]
                    )
                else:
                    writer.writerow([f"C{c}"] + ["N/A"] * num_heads + ["N/A"])

    print(f"Exported {num_layers * 2} CSV files to {output_dir}")

    # Summary CSV
    summary_file = os.path.join(output_dir, "summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "chunk", "compute_density_pct", "kv_density_pct"])
        for layer in range(num_layers):
            for c in chunks:
                if c in stats.get(layer, {}):
                    cd, kv = stats[layer][c]
                    writer.writerow([layer, c, f"{cd:.2f}", f"{kv:.2f}"])

    print(f"Exported summary to {summary_file}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <log_file> <output_dir>")
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = sys.argv[2]

    stats, kv_heads, comp_heads = parse_log(log_file)

    if not stats:
        print("No BLASST density data found in log file.")
        sys.exit(1)

    export_csvs(stats, kv_heads, comp_heads, output_dir)


if __name__ == "__main__":
    main()
