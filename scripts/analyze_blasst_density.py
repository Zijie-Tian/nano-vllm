#!/usr/bin/env python3
"""
Analyze BLASST density logs and print summary tables to stdout.

Produces:
  - Compute Density table (layer × chunk)
  - KV Density table (layer × chunk)
  - Layer group averages (Early/Middle/Late-Mid/Late)
  - Per-chunk averages
  - Top densest/sparsest layers
  - Per-head density analysis

Usage:
    python scripts/analyze_blasst_density.py <log_file>

Example:
    python scripts/analyze_blasst_density.py /tmp/blasst.log
"""

import re
import sys
from collections import defaultdict


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not log_file:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)

    # Patterns
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
                vals = re.findall(r"H\d+=([\d.]+)%", m.group(3))
                kv_heads[layer][chunk] = [float(v) for v in vals]
                continue
            m = comp_head_pattern.search(line)
            if m:
                layer, chunk = int(m.group(1)), int(m.group(2))
                vals = re.findall(r"H\d+=([\d.]+)%", m.group(3))
                comp_heads[layer][chunk] = [float(v) for v in vals]
                continue

    if not stats:
        print("No BLASST density data found in log file.")
        sys.exit(1)

    num_layers = max(stats.keys()) + 1
    chunks = sorted(set(c for l in stats.values() for c in l.keys()))
    num_heads = 0
    if kv_heads:
        first_layer = next(iter(kv_heads.values()))
        first_chunk = next(iter(first_layer.values()))
        num_heads = len(first_chunk)

    print(f"{'=' * 80}")
    print(f"BLASST Density Analysis")
    print(f"{'=' * 80}")
    print(f"Layers: {num_layers}, Chunks: {len(chunks)}, Heads: {num_heads}")
    print()

    # ====== Table 1: Compute Density ======
    print(f"{'=' * 80}")
    print("Table 1: Compute Density (%) per Layer per Chunk")
    print(f"{'=' * 80}")
    hdr = f"{'Layer':>6}"
    for c in chunks:
        hdr += f"  {'C' + str(c):>7}"
    hdr += f"  {'Avg':>7}"
    print(hdr)
    print("-" * len(hdr))

    layer_avg_cd = {}
    for layer in range(num_layers):
        row = f"{layer:>6}"
        vals = []
        for c in chunks:
            if c in stats[layer]:
                cd = stats[layer][c][0]
                row += f"  {cd:>7.2f}"
                vals.append(cd)
            else:
                row += f"  {'N/A':>7}"
        avg = sum(vals) / len(vals) if vals else 0
        layer_avg_cd[layer] = avg
        row += f"  {avg:>7.2f}"
        print(row)

    print()

    # ====== Table 2: KV Density ======
    print(f"{'=' * 80}")
    print("Table 2: Required KV Density (%) per Layer per Chunk")
    print(f"{'=' * 80}")
    hdr = f"{'Layer':>6}"
    for c in chunks:
        hdr += f"  {'C' + str(c):>7}"
    hdr += f"  {'Avg':>7}"
    print(hdr)
    print("-" * len(hdr))

    layer_avg_kv = {}
    for layer in range(num_layers):
        row = f"{layer:>6}"
        vals = []
        for c in chunks:
            if c in stats[layer]:
                kv = stats[layer][c][1]
                row += f"  {kv:>7.2f}"
                vals.append(kv)
            else:
                row += f"  {'N/A':>7}"
        avg = sum(vals) / len(vals) if vals else 0
        layer_avg_kv[layer] = avg
        row += f"  {avg:>7.2f}"
        print(row)

    print()

    # ====== Summary: Layer groups ======
    print(f"{'=' * 80}")
    print("Summary: Layer Group Average Densities")
    print(f"{'=' * 80}")

    quarter = num_layers // 4
    groups = [
        (f"Early (0-{quarter - 1})", range(0, quarter)),
        (f"Middle ({quarter}-{2 * quarter - 1})", range(quarter, 2 * quarter)),
        (
            f"Late-Mid ({2 * quarter}-{3 * quarter - 1})",
            range(2 * quarter, 3 * quarter),
        ),
        (f"Late ({3 * quarter}-{num_layers - 1})", range(3 * quarter, num_layers)),
    ]

    for name, rng in groups:
        cd_vals = [layer_avg_cd[l] for l in rng if l in layer_avg_cd]
        kv_vals = [layer_avg_kv[l] for l in rng if l in layer_avg_kv]
        avg_cd = sum(cd_vals) / len(cd_vals) if cd_vals else 0
        avg_kv = sum(kv_vals) / len(kv_vals) if kv_vals else 0
        print(f"  {name:>25}: Compute={avg_cd:.2f}%, KV Density={avg_kv:.2f}%")

    print()

    # ====== Per-chunk analysis ======
    print(f"{'=' * 80}")
    print("Summary: Per-Chunk Average Densities (across all layers)")
    print(f"{'=' * 80}")
    for c in chunks:
        cd_vals = [stats[l][c][0] for l in range(num_layers) if c in stats[l]]
        kv_vals = [stats[l][c][1] for l in range(num_layers) if c in stats[l]]
        avg_cd = sum(cd_vals) / len(cd_vals) if cd_vals else 0
        avg_kv = sum(kv_vals) / len(kv_vals) if kv_vals else 0
        print(f"  Chunk {c}: Compute={avg_cd:.2f}%, KV Density={avg_kv:.2f}%")

    print()

    # ====== Top densest/sparsest ======
    n_top = min(10, num_layers)
    print(f"{'=' * 80}")
    print(f"Top {n_top} Densest Layers (by avg KV Density)")
    print(f"{'=' * 80}")
    sorted_layers = sorted(layer_avg_kv.items(), key=lambda x: x[1], reverse=True)
    for layer, kv in sorted_layers[:n_top]:
        print(
            f"  Layer {layer:>2}: KV={kv:.2f}%, Compute={layer_avg_cd[layer]:.2f}%"
        )

    print()
    print(f"{'=' * 80}")
    print(f"Top {n_top} Sparsest Layers (by avg KV Density)")
    print(f"{'=' * 80}")
    for layer, kv in sorted_layers[-n_top:]:
        print(
            f"  Layer {layer:>2}: KV={kv:.2f}%, Compute={layer_avg_cd[layer]:.2f}%"
        )

    print()

    # ====== Head analysis ======
    head_data = kv_heads if kv_heads else comp_heads
    if head_data and num_heads > 0:
        print(f"{'=' * 80}")
        print("Head Analysis: Per-Head Average KV Density (all layers, all chunks)")
        print(f"{'=' * 80}")
        head_totals = [0.0] * num_heads
        head_counts = [0] * num_heads
        for layer in range(num_layers):
            for c in chunks:
                if c in kv_heads.get(layer, {}):
                    for h in range(num_heads):
                        head_totals[h] += kv_heads[layer][c][h]
                        head_counts[h] += 1

        head_avgs = [
            (h, head_totals[h] / head_counts[h] if head_counts[h] > 0 else 0)
            for h in range(num_heads)
        ]
        head_avgs_sorted = sorted(head_avgs, key=lambda x: x[1], reverse=True)

        print("  Top 5 densest heads:")
        for h, avg in head_avgs_sorted[:5]:
            print(f"    H{h}: {avg:.1f}%")
        print("  Top 5 sparsest heads:")
        for h, avg in head_avgs_sorted[-5:]:
            print(f"    H{h}: {avg:.1f}%")

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
