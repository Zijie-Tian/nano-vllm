---
description: Run the RULER results summarizer to aggregate all test results
---

This workflow aggregates the RULER benchmark evaluation results across all tested context lengths and methods into a single summary CSV file. It iterates through the `eval/RULER/scripts/benchmark_root` directory.

You can optionally append `--model` or `--method` arguments to filter the results, e.g., `python .agents/skills/ruler_results_summarizer/scripts/aggregate_ruler.py --model llama3.1-8b-nanovllm`.

1. Execute the Python script to aggregate the summary.csv results:
// turbo
```bash
python .agents/skills/ruler_results_summarizer/scripts/aggregate_ruler.py
```

2. Check the output at `results/ruler/ruler_summary.csv` to ensure it was created correctly.
