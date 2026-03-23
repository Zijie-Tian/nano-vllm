---
name: ruler_results_summarizer
description: Aggregates RULER benchmark results from summary.csv files into a single master summary table showing average task scores by context length.
---

# RULER Results Summarizer

This skill parses output directories produced by the RULER benchmark under `eval/RULER/scripts/benchmark_root/`. Specifically, it extracts the performance of various methods (e.g. baseline, Minference, COMPASS, BLASST, XAttention) across the 13 tasks and averages them. 

The collected averages are pivoted into a table formatted similarly to typical research papers:
* Rows: `Method_Model` (e.g., `BLASST(lambda=0.001) [glm4-9b-nanovllm]`)
* Columns: `Length` (e.g., 4k, 8k, 16k, 32k, 64k, 128k)
* Additional Column: `Avg.` (average across context lengths)
Missing values are represented as `-`.

## Usage

Simply run the aggregation script. It will automatically sweep the default `benchmark_root` folder and save the combined results to `results/ruler/ruler_summary.csv`.

```bash
python .agents/skills/ruler_results_summarizer/scripts/aggregate_ruler.py
```

Options available:
* `--root_dir`: The root folder to scan for `summary.csv` files (default: `eval/RULER/scripts/benchmark_root`)
* `--output_path`: The destination CSV file to write to (default: `results/ruler/ruler_summary.csv`)
* `--model`: Optional filter by exact base model name (e.g., `llama3.1-8b-nanovllm` or `glm4-9b-nanovllm`). If filtered, the model name suffix is omitted from row labels.
* `--method`: Optional filter by exact method abbreviation (e.g., `blasst`, `compass`, `xattn`).
