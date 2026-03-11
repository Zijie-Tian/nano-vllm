# COMPASS TMAC Verification Design & Data Pipeline

**Status**: 🟢 Active
**Component**: `COMPASSPolicy`, `verify_tmac_offload_accuracy.py`

---

## 1. Overview
This document outlines the design decisions and data pipeline used to verify the accuracy of the **T-MAC 2-bit Coarse Predictor** when integrated with the `COMPASSPolicy` for CPU-GPU Heterogeneous Offloading.

The goal is to ensure that the K-cache packaged into T-MAC's specific bit-serial interleaved format during the `offload_prefill_chunk` phase maintains acceptable numerical accuracy (NMSE < 0.2) when compared to a standard FP16 GEMM `Q @ K^T`.

## 2. The Verification Data Pipeline

During chunked prefill, the `COMPASSPolicy` must offload specific metadata to pinned CPU memory to allow the CPU to perform asynchronous, multiplication-free sparse predictions.

The required metadata includes:
1. **Query (Q) Chunk**: Saved during `select_blocks` into `_q_buffer`.
2. **Packed 2-bit K-Cache**: Quantized and packed during `offload_prefill_chunk` into `_k_packed_buffer`.

### 2.1 The Challenge: Sequence Deallocation & CPU Cache Reset
Initially, the verification script attempted to compare the output of `Q @ K_packed` against the standard `Q @ k_cache_cpu` (the primary CPU offload buffer managed by `OffloadEngine`). 

However, this caused the verification script to fail with an NMSE of `NaN`, because the K-cache read from CPU was all zeros.

**Root Cause:**
- In the `nanovllm` framework, when a sequence reaches EOS or maximum token length, `scheduler.postprocess` marks it as `FINISHED` and calls `kvcache_manager.deallocate(seq)`.
- Inside `HybridManager.deallocate`, after freeing the sequence's blocks, the system invokes `offload_engine.reset()`.
- To prevent cross-request state leakage (a critical bug fixed previously, see `docs/ruler_32k_chunked_offload_issue.md`), `OffloadEngine.reset()` explicitly executes `self.k_cache_cpu.zero_()`.
- Therefore, by the time the evaluation script finishes generation and triggers the verification hook, the actual CPU K-cache is completely wiped.

### 2.2 The Solution: The Bypass Verify Buffer (`_k_fp16_verify_buffer`)
To solve this without modifying the safety mechanisms of the core inference engine, we introduced a dedicated, bypassed tracking buffer inside `COMPASSPolicy`: `_k_fp16_verify_buffer`.

**Mechanism:**
1. In `alloc_policy_metadata`, alongside `_q_buffer` and `_k_packed_buffer`, we allocate `_k_fp16_verify_buffer`.
2. During `offload_prefill_chunk`, exactly when the raw FP16 K-cache is sent to be quantized and packed, a copy of that raw FP16 chunk is synchronously saved into `_k_fp16_verify_buffer`.
3. When the `test_ruler.py` run finishes, the evaluation hook reads from `_k_fp16_verify_buffer` (which is safely isolated from `OffloadEngine.reset()`).

## 3. Verification Execution
The test script `tests/verify_tmac_offload_accuracy.py` automatically hooks into `test_ruler.py` when `--sparse-policy COMPASS` is utilized.

**Verification Steps:**
1. Loads the actual raw Q chunk from `_q_buffer`.
2. Loads the actual raw FP16 K chunk from `_k_fp16_verify_buffer`.
3. Performs the exact 2-bit quantization and TMAC dequantization pipeline on the raw K data.
4. Computes `attn_ref = Q @ K_fp16^T`.
5. Computes `attn_dq = Q @ K_dequantized^T`.
6. Calculates the Normalized Mean Square Error (NMSE).

*Acceptance Criteria:* NMSE must remain < 0.20 to pass. 
*(Current baseline validation routinely achieves ~0.014 on realistic model activation data).*
