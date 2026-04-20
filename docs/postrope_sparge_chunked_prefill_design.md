# POSTROPE Sparge-Style Chunked Prefill Design

This document records the **current** `POSTROPE` design in `nano-vllm` after the SpargeAttention-inspired refactor work on 2026-04-19.

It is intentionally narrower than the original `docs/rope_policy_design.md` semantic note:
- `docs/rope_policy_design.md` explains the historical PREROPE / POSTROPE semantic split.
- **This document explains the current sparse `POSTROPE` implementation**, the design constraints that were explicitly chosen for chunked prefill, the validated RULER commands, and the current alignment / mismatch against the official SpargeAttn repository.

---

## 1. Scope and design constraints

The current `POSTROPE` implementation is a **SpargeAttn-style chunked-prefill adaptation** with the following hard constraints:

1. **Keep the public policy name `POSTROPE` unchanged.**
2. **Keep `apply_rope_in_attention = False`** and therefore preserve post-RoPE KV semantics.
3. **Implement `POSTROPE` as an independent policy** at the policy level:
   - no direct delegation to `FullAttentionPolicy`
   - no direct delegation to `BLASSTPolicy`
   - no imports from `nanovllm.ops.*` helper wrappers used as policy shortcuts
4. **Decode remains dense**. Sparse adaptation is only for chunked prefill.
5. **Do not add quantization-specific machinery** from SageAttention / SageAttention2 / SageAttention2++.
6. **Adapt to chunked prefill with a 1D selected-block interface**:
   - the implementation returns a 1D `selected_blocks: List[int]`
   - it does **not** attempt to reproduce the paper / official repo’s full 2D `M_g[i,j]` runtime interface
7. **Keep the normal physical KV chunk size at 4096** for the mainline validated path.

---

## 2. Current implementation status

### 2.1 Policy-level independence

The current `POSTROPE` policy is self-contained inside:

- [`nanovllm/kvcache/sparse/postrope.py`](../nanovllm/kvcache/sparse/postrope.py)

It implements locally:
- `select_blocks(...)`
- `compute_prefill(...)`
- `compute_decode(...)`
- `compute_chunked_prefill(...)`
- `compute_chunked_decode(...)`
- `on_prefill_offload(...)`

It also includes local helper implementations for:
- stage-2 Triton sparse prefill wrapper
- local attention-with-LSE wrapper
- local output/LSE merge logic

### 2.2 What “independent” means here

`POSTROPE` is independent **at the policy level**, but it is **not** a fully standalone runtime stack.

It still relies on low-level infrastructure such as:
- FlashAttention kernels (`flash_attn_varlen_func`, `flash_attn_with_kvcache`, `flash_attn_func`)
- Triton kernels embedded in `postrope.py`
- `OffloadEngine` ring-buffer / stream / slot-management APIs
- PyTorch / CUDA runtime

So the correct wording is:

> `POSTROPE` is a **policy-level independent implementation**, not a from-scratch reimplementation of the entire attention runtime.

---

## 3. Current algorithm flow

## 3.1 High-level structure

`POSTROPE` currently follows a two-stage sparse prefill structure:

1. **Stage 1 (`select_blocks`)**
   - GPU-side pooled-Q / pooled-K historical block selection
   - self-similarity-based fix-row / fix-column protection
   - returns a **1D list of selected historical KV parent blocks**

2. **Stage 2 (`compute_chunked_prefill`)**
   - exact-score chunked attention on the selected historical blocks plus the current chunk
   - online softmax-aware dynamic skipping implemented in a local Triton kernel

Decode is deliberately kept dense.

---

## 3.2 Stage 1 details

### Estimation granularity

The current implementation uses:

- `ESTIMATE_CHUNK_SIZE = 128`

This is only for **estimation / pooling / self-similarity analysis**.
The mainline validated offload / communication granularity remains:

- `kvcache_block_size = 4096`

### Pooled summaries

For stage 1, `POSTROPE` computes:
- pooled query summaries from the current prefill chunk
- pooled K summaries from historical CPU-offloaded blocks cached by `on_prefill_offload(...)`

The current helper methods are:
- `_mean_pool(...)`
- `_compute_self_cosine(...)`
- `_fold_q_heads(...)`
- `_fold_q_head_mask_any(...)`

### Current self-similarity semantics

After cross-checking the official SpargeAttn repository, the current implementation uses the following **repo-closer boolean semantics**:

#### K-side persistent / fix-column semantics
A historical parent KV chunk becomes persistent if:
- **any** KV head in that chunk has self-cosine below the active threshold

This is a chunked-prefill adaptation of the official repo logic:
- official repo computes boolean `sim_kblocks`
- then applies `final_map[~sim_kblocks] = 1`

#### Q-side persistent / fix-row semantics
A KV-head group becomes a Q-side fix-row head if:
- any grouped query head has a pooled query block whose self-cosine falls below the active threshold

This is implemented as:
1. threshold query self-cosine **per Q head first**
2. fold to KV-head groups via logical **OR**
3. if a KV-head group is triggered, all historical blocks are forced on for that head

This is the closest adaptation to the official repo that still respects the current 1D `selected_blocks` interface.

### Top-p selection

For heads not already forced on by fix-row / fix-column logic:
- pooled Q–K scores are computed
- persistent K chunks are masked out of the top-p competition
- `softmax` is computed over historical chunks
- cumulative top-p selection uses:
  - `TOP_P = 0.90`

### Additional runtime-specific keep rules

The current implementation also keeps:
- the first historical chunk as an anchor / attention-sink-style keep rule
- the most recent historical blocks according to:
  - `FORCE_RECENT_BLOCKS = 1`
  - `RECENT_TOKENS_BUDGET = 4096`

These are **engineering adaptations for the nano-vllm offload runtime** and are not part of the paper’s original abstract algorithm statement.

---

## 3.3 Stage 2 details

Stage 2 is a local Triton implementation embedded in `postrope.py`.

Its behavior is:
1. load selected historical KV parent blocks through `OffloadEngine`
2. compute exact `QK^T` scores block by block
3. maintain a running `m_global`
4. compare local row maxima against the running maximum
5. skip negligible blocks if their contribution is too small under the current softmax state
6. merge block outputs with local LSE merging logic

### Current thresholding rule

The current code uses:
- `FIXED_LAMBDA = 1e-4`
- therefore `ln(lambda) ≈ -9.2103`

The local Triton kernel compares:
- `max_diff = max(m_local - m_global)`
- and skips a block if:
  - `max_diff < ln(lambda)`

This is equivalent in spirit to the official SpargeAttn stage-2 “online exact-score filtering” idea, but expressed in the current local kernel parameterization.

### No quantization

Unlike the official SpargeAttn implementation paths in `spas_sage_attn/core.py`, the current `POSTROPE` stage 2 does **not** use:
- Q/K int8 quantization
- FP8 V quantization
- SageAttention / SageAttention2 / SageAttention2++ kernels

This is an explicit design choice.

---

## 3.4 Decode path

The current `POSTROPE` design keeps decode dense on purpose.

That means:
- no sparse block prediction for decode
- no Q-side / K-side sparse decode mask logic
- no quantization-specific decode machinery

This is intentional and matches the current project preference:

> adapt Sparge-style ideas to **chunked prefill first**, while keeping decode conservative and semantically dense.

---

## 4. Alignment against the official SpargeAttn repo

## 4.1 What is aligned

The current `POSTROPE` matches the official SpargeAttn repo at a high level in the following ways:

1. **two-stage structure**
   - stage 1 block selection
   - stage 2 online exact-score-aware skip
2. **mean pooling for coarse estimation**
3. **self-similarity-based fix-row / fix-column protection**
4. **top-p / cumulative-probability-style historical selection**
5. **stage-2 skip based on local max vs running global max**

## 4.2 What is intentionally different

The current `POSTROPE` is **not** a strict paper-faithful or repo-faithful full reproduction. Important differences are:

1. **1D selected-block interface instead of 2D `M_g[i,j]`**
   - this is the biggest structural deviation
   - it exists because the current offload runtime loads parent KV blocks, not arbitrary 2D block masks
2. **no quantization path**
3. **decode kept dense**
4. **offload-specific anchor / recent-block heuristics**
5. **fixed hyperparameters rather than per-layer calibrated search**

## 4.3 Current practical consequence of the 1D adaptation

When Q-side fix-row logic is implemented in a repo-closer way, the current 1D interface becomes noticeably more conservative:
- a triggered Q-side fix-row head often forces **all historical blocks** to remain selected for that head
- after union across heads, this reduces stage-1 sparsity

This is currently the main design tension in the `POSTROPE` adaptation.

---

## 5. Current validated results snapshot

The table below records the current validated milestones relevant to the active `POSTROPE` design.

| Iteration | Key change | RULER gate | Total time | Prefill total | Phase-1 density | Notes |
|---|---|---:|---:|---:|---:|---|
| 4 | policy-level independence, dense decode retained | 5/5 | 60.80s | 25.060s | 85.79% | no direct policy delegation |
| 6 | first Q-side fix-row adaptation (mean-before-grouping) | 5/5 | 64.66s | 28.763s | 85.79% | Q-side path did not trigger on the validated five samples |
| 7 | repo-closer boolean threshold-before-grouping semantics | 5/5 | 64.40s | 28.737s | 92.11% | Q-side path now triggers strongly; stage-1 becomes more conservative |

### Current phase-1 snapshot (validated 4096 mainline, Iteration 7)

- available blocks: `380`
- selected blocks: `350`
- density: `92.11%`
- persistent K blocks: `140`
- `q_persistent_groups_total = 2910`
- `q_force_heads_total = 95`
- `q_force_keep_all_calls = 65`

Interpretation:
- correctness remains good on the current gate
- but the repo-closer Q-side logic substantially reduces stage-1 sparsity under the current 1D interface

---

## 6. Canonical RULER test commands for the current POSTROPE path

## 6.1 Mainline smoke test (GPU1, 4096 block size)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH \
python tests/test_ruler.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --data-dir tests/data/ruler_32k \
  --datasets niah_single_1 \
  --sample-indices 0 \
  --max-model-len 40960 \
  --enable-offload \
  --sparse-policy POSTROPE \
  --quiet --json-output
```

## 6.2 Mainline 5-sample validation gate (GPU1, 4096 block size)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH \
python tests/test_ruler.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --data-dir tests/data/ruler_32k \
  --datasets niah_single_1 \
  --sample-indices 0,1,2,3,4 \
  --max-model-len 40960 \
  --enable-offload \
  --sparse-policy POSTROPE \
  --quiet --json-output
```

## 6.3 Optional 1024-block communication experiment

This is **not** the mainline recommended path, but it is kept here because it was validated and is useful for communication-density experiments.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/tzj/Code/nano-vllm:$PYTHONPATH \
python tests/test_ruler.py \
  --model /mnt/data/tzj/models/Llama-3.1-8B-Instruct \
  --data-dir tests/data/ruler_32k \
  --datasets niah_single_1 \
  --sample-indices 0,1,2,3,4 \
  --max-model-len 40960 \
  --enable-offload \
  --sparse-policy POSTROPE \
  --block-size 1024 \
  --quiet --json-output
```

Notes:
- this `1024`-block experiment remained correct (`5/5`)
- but it was much slower end-to-end than the validated `4096` mainline

---

## 7. Open design question

The main unresolved issue is now:

> how to preserve the official SpargeAttn-style Q-side fix-row semantics while still getting useful stage-1 sparsity under the current 1D selected-block interface.

That is, the remaining problem is **not** whether Q-side fix rows should exist—they now do.
The problem is how to adapt them to the current chunked-prefill runtime without allowing them to collapse sparsity too often.

---

## 8. References

### External references
- SpargeAttention paper (arXiv abstract): https://arxiv.org/abs/2502.18137
- SpargeAttention paper (PDF): https://arxiv.org/pdf/2502.18137
- Official SpargeAttn repository: https://github.com/thu-ml/SpargeAttn
- Official SpargeAttn repo `utils.py` (stage-1 block-map construction): https://github.com/thu-ml/SpargeAttn/blob/ae5b629/spas_sage_attn/utils.py
- Official SpargeAttn repo `core.py` (current wrapper / backend path): https://github.com/thu-ml/SpargeAttn/blob/ae5b629/spas_sage_attn/core.py
- Official SpargeAttn repo stage-2 CUDA kernel example (`pv_threshold` logic): https://github.com/thu-ml/SpargeAttn/blob/ae5b629/csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh

### Internal implementation references
- Current `POSTROPE` implementation: [`nanovllm/kvcache/sparse/postrope.py`](../nanovllm/kvcache/sparse/postrope.py)
- RULER test entrypoint: [`tests/test_ruler.py`](../tests/test_ruler.py)
- Historical semantic note: [`docs/rope_policy_design.md`](rope_policy_design.md)
- General `test_ruler.py` usage guide: [`docs/test_ruler_usage_guide.md`](test_ruler_usage_guide.md)
