# Chunked Prefill Timing Sync Default-Off Note

This note records the change that makes `NANOVLLM_CHUNKED_TIMING_SYNC` default to **off** in `nanovllm/layers/attention.py`.

## What changed

The chunked-prefill timing gate now uses:

```python
TIMING_SYNC_ENABLED = os.environ.get("NANOVLLM_CHUNKED_TIMING_SYNC", "0") == "1"
```

That means:

- default behavior: **timing sync disabled**
- opt-in legacy behavior: set `NANOVLLM_CHUNKED_TIMING_SYNC=1`

## Why

`ChunkedPrefillTimer` previously forced `torch.cuda.current_stream().synchronize()` around:

1. pre-`select_blocks`
2. pre-`compute_chunked_prefill`
3. post-`compute_chunked_prefill`

Those synchronizations perturb the runtime and reduce overlap, especially in chunked offload mode.

The default-off change keeps the timer available as a debugging/profiling tool while preventing it from imposing synchronization overhead in normal runs.

## Behavioral impact

### Model behavior

No intended inference-logic change.

This toggle only changes whether the timer inserts explicit stream synchronization for measurement boundaries.

### Timer semantics

With timing sync disabled:

- the model/runtime behavior is less perturbed
- overlap between phases is preserved better
- per-phase timer numbers are **not directly comparable** to the legacy synchronized timing breakdown

So the timer remains useful for rough observability, but it is no longer a strict phase-isolation measurement unless `NANOVLLM_CHUNKED_TIMING_SYNC=1` is set explicitly.

## Validation

Validation was rerun on **GPU1** using the existing RULER validator workload:

- model: `~/models/Llama-3.1-8B-Instruct`
- data: `tests/data/ruler_32k`
- task: `niah_single_1`
- samples: `0,1,2,3,4`
- offload enabled
- block size: `4096`
- num GPU blocks: `4`

Observed result:

- `niah_single_1`: **5 / 5**
- accuracy: **100%**

So the default-off change preserves the current RULER pass behavior.

## How to re-enable synchronized timing

Use:

```bash
NANOVLLM_CHUNKED_TIMING_SYNC=1
```

Example:

```bash
NANOVLLM_CHUNKED_TIMING_SYNC=1 \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH \
python tests/test_ruler.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --data-dir tests/data/ruler_32k \
  --datasets niah_single_1 \
  --sample-indices 0 \
  --max-model-len 40960 \
  --enable-offload \
  --block-size 4096 \
  --num-gpu-blocks 4 \
  --sparse-policy POSTROPE \
  --json-output --quiet
```
