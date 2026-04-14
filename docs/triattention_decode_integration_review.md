# TriAttention decode integration review memo

Date: 2026-04-14
Owner: worker-4 (task 7 integration/review lane)

## Highest-signal invariants

1. **Pre-RoPE Q/K must remain the active TriAttention contract.**
   - `nanovllm/layers/attention.py:170-179` enables lazy-RoPE mode via `apply_rope_in_attention`.
   - `nanovllm/models/llama.py:79-90`, `nanovllm/models/qwen2.py:82-93`, `nanovllm/models/qwen3.py:96-107`, and `nanovllm/models/glm4.py:84-95` all skip model-side RoPE when that flag is set.
   - `nanovllm/kvcache/sparse/triattention.py:48-88` then becomes the only place that may rotate Q/K before attention. Any sparse-decode work that stores post-RoPE K in cache/offload buffers will double-rotate or desynchronize prefill/decode semantics.

2. **`selected_blocks` is not just a set; it is an ordered timeline.**
   - `nanovllm/kvcache/sparse/policy.py:245-277` documents `select_blocks()` as ordered CPU block ids.
   - `nanovllm/layers/attention.py:471-500` forwards that list directly into chunked decode.
   - `nanovllm/kvcache/sparse/triattention.py:676-685` derives RoPE positions from `block_idx`, not from `cpu_block_id`.
   - Consequence: any sparse selector must sort retained blocks back into original sequence order before compute. A top-k list returned in score order will silently assign wrong absolute positions.

3. **Partial-block handling assumes the chronologically last retained prefill block stays last.**
   - `nanovllm/kvcache/sparse/triattention.py:533-548` only treats the final selected block as potentially partial.
   - `_decode_ring_buffer_pipeline()` then truncates only `block_idx == num_blocks - 1` at `nanovllm/kvcache/sparse/triattention.py:666-675`.
   - If sparse decode keeps the real tail block but reorders it earlier, or if it introduces compaction metadata without updating this logic, token counts and positions will drift.

4. **GPU-only decode has the same ordering hazard as offload decode.**
   - `nanovllm/kvcache/sparse/triattention.py:281-349` iterates paged cache blocks sequentially and also assigns positions from loop index (`block_start = block_idx * block_size`).
   - If GPU-only sparse decode is added, it cannot just shrink the block table; it must preserve chronological order or compute positions from original cache offsets.

## Merge/conflict hotspots

1. **TriAttention and Full policy decode paths are mirrored.**
   - `nanovllm/kvcache/sparse/full_policy.py:320-562`
   - `nanovllm/kvcache/sparse/triattention.py:501-716`
   - Most structural edits to decode/offload pipeline will conflict in both places. Keep functional fixes narrow and port them intentionally rather than letting the two files drift.

2. **TriAttention prefill GPU path is a lazy-RoPE fork of the full policy baseline.**
   - `nanovllm/kvcache/sparse/full_policy.py:88-149`
   - `nanovllm/kvcache/sparse/triattention.py:169-349`
   - The split between history (`causal=False`) and current chunk (`causal=True`) is correct for current semantics; sparse decode changes should avoid reworking this prefill path unless absolutely required.

3. **The current generic policy interface may be too weak for external TriAttention semantics.**
   - `nanovllm/kvcache/sparse/policy.py:21-154` already has richer selection containers (`SubBlockSelection`, `PerHeadSubBlockSelection`, `TensorSelection`).
   - But `select_blocks()` still returns `List[int]` and the attention caller assumes `len(selected_blocks)` / direct iteration.
   - If worker-1 upgrades TriAttention from whole-block selection toward token/per-head compaction, `attention.py`, `triattention.py`, and generic logging will need a synchronized interface change.

## Concrete pitfalls for sparse decode integration

1. **`PolicyContext.total_kv_len` is only an approximation in decode.**
   - `nanovllm/layers/attention.py:477-485` sets it to `len(cpu_block_table) * block_size`, not the true prefill length.
   - Budget/window logic should prefer `kvcache_manager.get_prefill_len(seq)` or explicit policy state instead of trusting this field.

2. **TriAttention config plumbing is currently missing.**
   - `nanovllm/config.py:12-17` still describes `TRIATTENTION` as “Full attention semantics, but apply RoPE inside Attention”.
   - `nanovllm/kvcache/__init__.py:49-85` and `:111-141` pass no TriAttention-specific kwargs.
   - External TriAttention decode needs at least stats/budget/window/divide-length style inputs, so config + CLI/help + tests will need coordinated updates.

3. **Chunked decode depends on the retained blocks being contiguous in logical time for RoPE reconstruction, but not contiguous in CPU block ids.**
   - `nanovllm/kvcache/sparse/triattention.py:648-706` loads arbitrary CPU block ids just fine.
   - The fragile part is only the position reconstruction (`_block_positions`).
   - Safe minimal path: preserve chronological order and map positions from original sequence offsets, not from dense post-selection index.

4. **Debug logs are still useful while integrating, but there are several noisy `[DEBUG]` sites that will become harder to read once sparse decode starts pruning aggressively.**
   - `nanovllm/kvcache/sparse/triattention.py:371-375`, `:523-527`
   - `nanovllm/layers/attention.py:489-497`
   - Keep them until correctness is stable, then downgrade or tighten them after merge.

## Suggested low-risk landing order

1. Extend config/factory plumbing for TriAttention-specific runtime knobs.
2. Add decode selection state/metadata without changing prefill semantics.
3. Make block-position reconstruction use original absolute offsets.
4. Only then consider richer per-head or token-level compaction types.

## Bottom line

The biggest correctness trap is **position reconstruction after sparse selection**. The current implementation assumes selected blocks remain in chronological order and infers absolute positions from dense loop indices. If sparse decode lands without fixing or preserving that invariant, outputs can look superficially valid while being semantically wrong.
