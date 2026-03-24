/**
 * qk_blockmask_fp32.cpp - FP32 naive reference for BLASST block-mask generation
 */

#include "cpu_ops/qk_blockmask.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cpu_ops {

template <int BS, int STEP_KV>
void qk_blockmask_fp32(const float* Q, const float* K,
                       float* block_rowmax,
                       float* running_max,
                       uint8_t* block_mask,
                       float log_lambda,
                       size_t BQ, size_t BK, size_t D) {
    size_t n_q_blocks = (BQ + BS - 1) / BS;
    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;

    for (size_t kblk = 0; kblk < n_k_blocks; ++kblk) {
        size_t k_start = kblk * STEP_KV;
        size_t k_end = std::min(k_start + (size_t)STEP_KV, BK);

        for (size_t i = 0; i < BQ; ++i) {
            float local_max = -std::numeric_limits<float>::infinity();
            for (size_t j = k_start; j < k_end; ++j) {
                float dot = 0.0f;
                for (size_t d = 0; d < D; ++d)
                    dot += Q[i * D + d] * K[j * D + d];
                local_max = std::max(local_max, dot);
            }
            block_rowmax[i * n_k_blocks + kblk] = local_max;
            running_max[i] = std::max(running_max[i], local_max);
        }

        for (size_t qblk = 0; qblk < n_q_blocks; ++qblk) {
            size_t q_start = qblk * BS;
            size_t q_end = std::min(q_start + (size_t)BS, BQ);
            bool all_skip = true;
            for (size_t i = q_start; i < q_end; ++i) {
                if ((block_rowmax[i * n_k_blocks + kblk] - running_max[i]) >= log_lambda) {
                    all_skip = false;
                    break;
                }
            }
            block_mask[qblk * n_k_blocks + kblk] = all_skip ? 0 : 1;
        }
    }
}

#define INSTANTIATE(BS, STEP_KV) \
    template void qk_blockmask_fp32<BS, STEP_KV>( \
        const float*, const float*, float*, float*, uint8_t*, \
        float, size_t, size_t, size_t);

INSTANTIATE(128, 128)
INSTANTIATE(128, 64)
INSTANTIATE(64, 128)
INSTANTIATE(64, 64)

#undef INSTANTIATE

}  // namespace cpu_ops
