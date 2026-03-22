# COMPASS L1 Selection — SpargeAttn Alignment

This document records the SpargeAttn-aligned L1 block selection implementation in COMPASS, including the algorithm, parameter sweep results, and testing instructions.

## Algorithm

COMPASS L1 selection now replicates the exact algorithm from [SpargeAttn](https://github.com/thu-ml/SpargeAttn) `get_block_map_meansim()`, adapted for CPU-side execution in a chunked prefill context.

### Steps

1. **Mean Pool**: Q and K are mean-pooled per 128-token sub-block (no L2 normalization).
2. **Scaled Dot-Product**: `scores = Q_pool @ K_pool^T * d^{-0.5}` → `[H, G_q, G_k]`.
3. **Self-Cosine Masking**: Diverse K blocks (self-cos < theta) are set to `-inf` in scores, excluding them from the softmax budget. They are force-selected later.
4. **Per-Q-Block Softmax**: `softmax(scores, dim=-1)` → `[H, G_q, G_k]`. Each Q block independently evaluates all K blocks.
5. **Per-Q-Block Top_p**: CDF-based selection per (head, Q-block) pair via `searchsorted`.
6. **Force-Select**: Diverse K blocks (`self_cos < theta`) and diverse Q blocks are force-selected into the final map.
7. **Attention Sink**: Block 0 is always selected (`final_map[:, :, 0] = True`).
8. **Union Across Q**: `final_map.any(dim=1)` → `[H, G_k]` for per-head IO decisions.

### Key Differences from Previous COMPASS Implementation

| Aspect | Previous COMPASS | SpargeAttn-Aligned |
|---|---|---|
| Pooling | L2-normalized | Raw mean pool |
| Scoring | No `d^{-0.5}` scaling | Scaled dot-product |
| Softmax | Average across Q → softmax `[H, G_k]` | Per-Q-block softmax `[H, G_q, G_k]` |
| Top_p | On averaged `[H, G_k]` | Per-Q-block `[H, G_q]` |
| Self-cos | Post-hoc force-select only | Mask before softmax + force-select |
| Sink | None | `mask[:, :, 0] = True` |

## Parameters

| CLI Argument | Config Name | Default | Description |
|---|---|---|---|
| `--compass-top-p` | `compass_top_p` | `0.9` | CDF threshold for per-Q-block selection. Lower → more aggressive pruning. |
| `--compass-theta` | `compass_theta` | `0.6` | Self-cosine threshold. Blocks with self-cos < theta are "diverse" and force-selected. Higher theta → more force-selected → less pruning. |
| `--compass-lambda` | `lambda_threshold` | `0.001` | L2 GPU dynamic pruning (BLASST-style). Set to `1e-10` to disable and isolate L1. |

## Parameter Sweep Results

**Setup**: Llama-3.1-8B-Instruct, 32k context, NIAH single-needle, 1 sample, `lambda=1e-10` (L2 disabled).

| top_p | theta | Accuracy | Selection Rate | L1 Pruning |
|---|---|---|---|---|
| 0.5 | 0.3 | ✅ 100% | 0.792 | **20.8%** |
| 0.5 | 0.6 | ✅ 100% | 0.801 | **19.9%** |
| 0.6 | 0.6 | ✅ 100% | 0.847 | **15.3%** |
| 0.7 | 0.3 | ✅ 100% | 0.883 | **11.7%** |
| 0.7 | 0.6 | ✅ 100% | 0.886 | **11.4%** |
| 0.8 | 0.6 | ✅ 100% | 0.923 | **7.7%** |
| 0.9 | 0.3 | ✅ 100% | 0.956 | **4.4%** |
| 0.9 | 0.6 | ✅ 100% | 0.957 | **4.3%** |
| 0.5 | 0.9 | ✅ 100% | 1.000 | 0.0% |
| 0.7 | 0.9 | ✅ 100% | 1.000 | 0.0% |
| 0.9 | 0.9 | ✅ 100% | 1.000 | 0.0% |

### Observations

1. **theta=0.9 disables pruning**: Nearly all sub-blocks have self-cos > 0.9, so `theta=0.9` marks almost everything as "diverse" → force-selected → top_p has no effect.
2. **theta=0.3 vs 0.6**: Minimal difference (~1% selection rate). Most blocks have self-cosine well above 0.6.
3. **top_p is the primary control**: `top_p=0.5` achieves ~20% L1 pruning while maintaining 100% NIAH accuracy.
4. **Per-Q union limits pruning**: Since each Q block independently selects K blocks and the final mask is their union, the overall pruning is bounded. Different Q blocks select different K blocks, keeping the union high.

## Testing Commands

All commands require `source build/nano-vllm-envs.sh` first.

### Correctness Baseline (100% selection, no pruning)

```bash
source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 --num-samples 1 \
    --max-model-len 40960 --enable-offload \
    --sparse-policy COMPASS \
    --compass-top-p 1.0 --compass-lambda 1e-10 --compass-theta 0.6
```

Expected: 100% accuracy, Selection rate = 1.000, L1 pruning = 0.0%.

### L1-Only Sparse Test (isolate L1 effect)

```bash
source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 --num-samples 1 \
    --max-model-len 40960 --enable-offload \
    --sparse-policy COMPASS \
    --compass-top-p 0.5 --compass-lambda 1e-10 --compass-theta 0.3
```

Expected: 100% accuracy, Selection rate ≈ 0.79, L1 pruning ≈ 20%.

### Full Sparse Test (L1 + L2)

```bash
source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 --num-samples 1 \
    --max-model-len 40960 --enable-offload \
    --sparse-policy COMPASS \
    --compass-top-p 0.9 --compass-lambda 0.0001 --compass-theta 0.6
```

Expected: 100% accuracy, L1 pruning ≈ 4%, additional L2 pruning via BLASST.

### Parameter Sweep Script

```bash
#!/bin/bash
source build/nano-vllm-envs.sh
MODEL=~/models/Llama-3.1-8B-Instruct
DATA=tests/data/ruler_32k
LAMBDA=1e-10  # disable L2

for THETA in 0.3 0.6 0.9; do
  for TOP_P in 0.5 0.7 0.9; do
    echo "=== top_p=$TOP_P theta=$THETA ==="
    CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py \
        --model $MODEL --data-dir $DATA \
        --datasets niah_single_1 --num-samples 1 \
        --max-model-len 40960 --enable-offload \
        --sparse-policy COMPASS \
        --compass-top-p $TOP_P --compass-lambda $LAMBDA --compass-theta $THETA 2>&1 \
        | grep -E "(Selection rate|Sub-block pruning|TOTAL)"
  done
done
```

## Log Interpretation

Example log line:
```
[COMPASS] layer=31, seq_chunk=7: IO_density=100.0% (7/7), L1_density=98.1% (1758/1792), Persistent_K=14.3% (1/7), per-head: [H0:100.0%, ...]
```

| Field | Meaning |
|---|---|
| `IO_density` | Percentage of KV cache blocks (1024 tokens each) needing H2D transfer (union across all heads and sub-blocks). |
| `L1_density` | Percentage of 128-token sub-blocks selected across all heads. `total_selected / (G_k × H)`. |
| `Persistent_K` | Blocks force-selected by self-cosine diversity (always transferred regardless of top_p). |
| `per-head` | Per-head L1 density breakdown. |

## Known Limitations

1. **Per-Q union bounds pruning**: The union across Q blocks limits maximum achievable L1 sparsity to ~20% on 32k context. Longer contexts or larger Q block sizes may enable more pruning.
2. **CPU top-p overhead**: The Python double loop for scatter (`fill_block_map`) takes ~2.4s (70% of CPU time). Can be optimized with vectorized tensor ops or a Triton CPU kernel.
3. **Single-sample test**: Results are based on 1 NIAH sample. Multi-sample and multi-task benchmarks are needed for comprehensive validation.
