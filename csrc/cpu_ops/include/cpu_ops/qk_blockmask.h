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
 *
 * Common parameters:
 *   input_mask  [n_q_blocks × n_k_blocks] nullable. 1=compute, 0=skip.
 *               When null, all blocks are computed.
 *               Allows external causal mask or arbitrary block-level masks.
 *   scale       Scaling factor applied to QK dot products (typically 1/sqrt(D)).
 *               Set to 1.0f for unscaled.
 */

#include <cstddef>
#include <cstdint>

namespace cpu_ops {

/**
 * FP32 reference for BLASST block-mask generation (naive, no SIMD).
 */
template <int BS = 128, int STEP_KV = 128>
void qk_blockmask_fp32(const float* Q, const float* K,
                       float* block_rowmax,
                       float* running_max,
                       uint8_t* block_mask,
                       const uint8_t* input_mask,
                       float scale,
                       float log_lambda,
                       size_t BQ, size_t BK, size_t D);

/**
 * FP32 BLASST block-mask generation (HPC-optimized, OpenMP parallel).
 */
template <int BS = 128, int STEP_KV = 128>
void qk_blockmask_fp32_omp(const float* Q, const float* K,
                           float* block_rowmax,
                           float* running_max,
                           uint8_t* block_mask,
                           const uint8_t* input_mask,
                           float scale,
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
                       const uint8_t* input_mask,
                       float scale,
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
                           const uint8_t* input_mask,
                           float scale,
                           float log_lambda,
                           size_t BQ, size_t BK, size_t D);

}  // namespace cpu_ops
