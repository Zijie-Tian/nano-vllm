#pragma once
/**
 * cpu_ops/qk_blockmask.h - BLASST block-level skip mask generation
 *
 * Template parameters:
 *   MR:      Q-row register blocking
 *   NR:      K-tile (×16) SIMD blocking
 *   GS:      VNNI INT8 quantization group size (0 = per-tensor)
 *   BS:      Q-dim voting window (AND-reduce across BS Q-rows)
 *   STEP_KV: K-dim skip tile granularity (one skip decision per STEP_KV K-tokens)
 */

#include <cstddef>
#include <cstdint>

namespace cpu_ops {

/**
 * FP32 reference for BLASST block-mask generation (naive, no SIMD).
 *
 * @param block_rowmax  [BQ × n_k_blocks] per-row per-K-tile local rowmax
 * @param running_max   [BQ] in/out: streaming global max
 * @param block_mask    [n_q_blocks × n_k_blocks] 1=keep, 0=skip (AND-reduced)
 * @param log_lambda    ln(λ) threshold
 */
template <int BS = 128, int STEP_KV = 128>
void qk_blockmask_fp32(const float* Q, const float* K,
                       float* block_rowmax,
                       float* running_max,
                       uint8_t* block_mask,
                       float log_lambda,
                       size_t BQ, size_t BK, size_t D);

/**
 * VNNI INT8 BLASST block-mask generation (single-threaded).
 */
template <int MR, int NR, int GS, int BS = 128, int STEP_KV = 128>
void qk_blockmask_vnni(const float* Q, const int8_t* K_vnni,
                       const float* scale_k, const int32_t* sum_k,
                       float* block_rowmax,
                       float* running_max,
                       uint8_t* block_mask,
                       float log_lambda,
                       size_t BQ, size_t BK, size_t D);

/**
 * VNNI INT8 BLASST block-mask generation (OpenMP parallel).
 */
template <int MR, int NR, int GS, int BS = 128, int STEP_KV = 128>
void qk_blockmask_vnni_omp(const float* Q, const int8_t* K_vnni,
                           const float* scale_k, const int32_t* sum_k,
                           float* block_rowmax,
                           float* running_max,
                           uint8_t* block_mask,
                           float log_lambda,
                           size_t BQ, size_t BK, size_t D);

}  // namespace cpu_ops
