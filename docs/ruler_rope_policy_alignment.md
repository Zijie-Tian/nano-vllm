# RULER Alignment Record for FULL / POSTROPE / PREROPE

This document records the exact-output alignment checks for the three full-attention semantic policies:

- `FULL`
- `POSTROPE`
- `PREROPE`

The focus is **offload + chunked prefill** behavior.

## Goal

Verify that `PREROPE` preserves the same inference semantics as `FULL`, and that `POSTROPE` remains an explicit alias of the same baseline behavior.

## Test environment

- **date**: 2026-04-19
- **environment**: `conda run -n compass`
- **model**: `~/models/Llama-3.1-8B-Instruct`
- **dataset**: `tests/data/ruler_32k`
- **task**: `niah_single_1`
- **samples**: `0,1,2,3,4`
- **GPU**: both `GPU0` and `GPU1`
- **offload**: enabled
- **chunked prefill**: enabled implicitly by the offload path for this long prompt
- **block size**: `4096`
- **num_gpu_blocks**: `4`
- **max_model_len / max_num_batched_tokens**: `40960`
- **sampling**: `temperature=0.1`, `max_tokens=16`

## Code path

The comparison reused the same loading and prompt-conversion logic as `tests/test_ruler.py`:

- `load_samples(...)`
- `convert_prompt_for_model(...)`
- `LLM.generate(...)`

This keeps the comparison anchored to the same prompt formatting and model execution path used by the benchmark harness.

## Why this is a chunked-prefill validation

For these samples, the prompt length is roughly 32K tokens. Combined with:

- `enable_cpu_offload=True`
- `kvcache_block_size=4096`
- `num_gpu_blocks=4`

the execution goes through the ring-buffer offload path and chunked prefill flow in `model_runner.py` and `attention.py`.

## Exact outputs

All three policies produced the same outputs on both GPUs.

| sample | expected | output |
|---|---|---|
| 0 | `8930103` | `: 8930103.<|eot_id|>` |
| 1 | `4194548` | `: 4194548.<|eot_id|>` |
| 2 | `8231838` | `: 8231838.<|eot_id|>` |
| 3 | `8835373` | `: 8835373.<|eot_id|>` |
| 4 | `7754864` | `: 7754864.<|eot_id|>` |

## Equality results

### Same-GPU policy comparison

For **GPU0**:

- `FULL == POSTROPE` (text and token IDs)
- `FULL == PREROPE` (text and token IDs)
- `POSTROPE == PREROPE` (implied by the above)

For **GPU1**:

- `FULL == POSTROPE` (text and token IDs)
- `FULL == PREROPE` (text and token IDs)
- `POSTROPE == PREROPE` (implied by the above)

### Cross-GPU comparison

For each policy independently:

- `GPU0 == GPU1` (text and token IDs)

So the final outcome is:

> `FULL`, `POSTROPE`, and `PREROPE` are exactly aligned for these five `niah_single_1` samples under offload + chunked prefill, and the results are stable across GPU0 and GPU1.

## Recommended command pattern

For aggregate task validation, the direct benchmark command remains:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH \
    conda run -n compass python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --sample-indices 0,1,2,3,4 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy <FULL|POSTROPE|PREROPE> \
    --json-output
```

For exact text/token parity comparisons, use a small harness that imports the helper functions from `tests/test_ruler.py` so the prompt normalization and execution path stay identical.

## Interpretation

These results support the intended semantic contract:

- `POSTROPE` is the explicit post-RoPE baseline
- `PREROPE` changes cache semantics, but not generation results

That means PREROPE is now suitable for semantic experiments where the cache representation changes but the expected outputs must remain locked to the FULL baseline.

## Related docs

- [`docs/rope_policy_design.md`](docs/rope_policy_design.md)
- [`docs/sparse_policy_architecture.md`](docs/sparse_policy_architecture.md)
- [`docs/test_ruler_usage_guide.md`](docs/test_ruler_usage_guide.md)
