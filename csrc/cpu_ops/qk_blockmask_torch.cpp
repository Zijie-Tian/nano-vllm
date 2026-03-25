/**
 * qk_blockmask_torch.cpp - PyTorch C++ extension for BLASST block-mask ops
 *
 * Exposes:
 *   - pack_k_vnni(K, GS) -> (K_vnni, scale_k, sum_k)
 *   - qk_blockmask_vnni_omp(Q, K_vnni, scale_k, sum_k, log_lambda, scale, input_mask, BS, STEP_KV)
 *   - qk_blockmask_fp32_omp(Q, K, log_lambda, scale, input_mask, BS, STEP_KV)
 */

#include <torch/extension.h>
#include "cpu_ops/qk_blockmask.h"
#include "cpu_ops/qk_rowmax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================
// pack_k_vnni: inline implementation (avoids needing qk_rowmax_vnni.cpp)
// ============================================================
static void pack_k_vnni_impl(const float* K, int8_t* K_vnni, float* scale_k,
                             int32_t* sum_k, size_t BK, size_t D, size_t GS) {
    size_t n_tiles = (BK + 15) / 16;
    size_t d_groups = (D + 3) / 4;
    size_t n_qgroups = (GS == 0) ? 1 : ((BK + GS - 1) / GS);
    size_t actual_gs = (GS == 0) ? BK : GS;

    std::vector<int8_t> K_int8(BK * D, 0);

    for (size_t g = 0; g < n_qgroups; ++g) {
        size_t j_start = g * actual_gs;
        size_t j_end = std::min(j_start + actual_gs, BK);

        float absmax = 0.0f;
        for (size_t j = j_start; j < j_end; ++j)
            for (size_t d = 0; d < D; ++d)
                absmax = std::max(absmax, std::fabs(K[j * D + d]));

        float scale = absmax / 127.0f;
        float inv_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        scale_k[g] = scale;

        for (size_t j = j_start; j < j_end; ++j) {
            int32_t row_sum = 0;
            for (size_t d = 0; d < D; ++d) {
                int val = (int)std::roundf(K[j * D + d] * inv_scale);
                val = std::max(-128, std::min(127, val));
                K_int8[j * D + d] = (int8_t)val;
                row_sum += val;
            }
            sum_k[j] = row_sum;
        }
    }
    for (size_t j = BK; j < BK + 16; ++j) sum_k[j] = 0;

    memset(K_vnni, 0, n_tiles * d_groups * 64);
    for (size_t t = 0; t < n_tiles; ++t) {
        size_t j_base = t * 16;
        size_t j_count = std::min((size_t)16, BK - j_base);
        for (size_t dg = 0; dg < d_groups; ++dg) {
            int8_t* dst = K_vnni + (t * d_groups + dg) * 64;
            for (size_t j = 0; j < j_count; ++j)
                for (size_t k = 0; k < 4 && (dg * 4 + k) < D; ++k)
                    dst[j * 4 + k] = K_int8[(j_base + j) * D + dg * 4 + k];
        }
    }
}

// ============================================================
// Torch wrapper: pack_k_vnni
// ============================================================
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_k_vnni_torch(torch::Tensor K, int64_t GS) {
    TORCH_CHECK(K.device().is_cpu(), "K must be a CPU tensor");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(K.dim() == 2, "K must be 2D [BK, D]");
    K = K.contiguous();

    int64_t BK = K.size(0), D = K.size(1);
    size_t buf_size = cpu_ops::pack_k_vnni_buffer_size(BK, D);
    size_t ng = cpu_ops::num_quant_groups(BK, GS);

    auto K_vnni = torch::zeros({(int64_t)buf_size}, torch::kInt8);
    auto scale_k = torch::zeros({(int64_t)ng}, torch::kFloat32);
    auto sum_k = torch::zeros({BK + 16}, torch::kInt32);

    pack_k_vnni_impl(K.data_ptr<float>(),
                     K_vnni.data_ptr<int8_t>(),
                     scale_k.data_ptr<float>(),
                     sum_k.data_ptr<int32_t>(),
                     BK, D, GS);

    return std::make_tuple(K_vnni, scale_k, sum_k);
}

// ============================================================
// Torch wrapper: qk_blockmask_vnni_omp
// ============================================================
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
qk_blockmask_vnni_omp_torch(torch::Tensor Q,
                            torch::Tensor K_vnni,
                            torch::Tensor scale_k,
                            torch::Tensor sum_k,
                            double log_lambda,
                            double scale,
                            c10::optional<torch::Tensor> input_mask_opt,
                            int64_t BS, int64_t STEP_KV) {
    TORCH_CHECK(Q.device().is_cpu(), "Q must be CPU");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D [BQ, D]");
    Q = Q.contiguous();
    K_vnni = K_vnni.contiguous();
    scale_k = scale_k.contiguous();
    sum_k = sum_k.contiguous();

    int64_t BQ = Q.size(0), D = Q.size(1);
    int64_t BK = sum_k.size(0) - 16;
    TORCH_CHECK(BK > 0, "Invalid sum_k size");

    int64_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    int64_t n_q_blocks = (BQ + BS - 1) / BS;

    // Validate input_mask if provided
    const uint8_t* input_mask_ptr = nullptr;
    torch::Tensor input_mask;
    if (input_mask_opt.has_value()) {
        input_mask = input_mask_opt.value().contiguous();
        TORCH_CHECK(input_mask.device().is_cpu(), "input_mask must be CPU");
        TORCH_CHECK(input_mask.dtype() == torch::kUInt8,
                     "input_mask must be uint8");
        TORCH_CHECK(input_mask.dim() == 2 &&
                     input_mask.size(0) == n_q_blocks &&
                     input_mask.size(1) == n_k_blocks,
                     "input_mask must be [n_q_blocks, n_k_blocks]");
        input_mask_ptr = input_mask.data_ptr<uint8_t>();
    }

    auto block_rowmax = torch::empty({BQ, n_k_blocks}, torch::kFloat32);
    auto running_max = torch::full({BQ}, -std::numeric_limits<float>::infinity(),
                                   torch::kFloat32);
    auto block_mask = torch::empty({n_q_blocks, n_k_blocks}, torch::kUInt8);

    auto call = [&](auto bs_tag, auto step_tag) {
        constexpr int bs_val = decltype(bs_tag)::value;
        constexpr int step_val = decltype(step_tag)::value;
        cpu_ops::qk_blockmask_vnni_omp<8, 2, 4096, bs_val, step_val>(
            Q.data_ptr<float>(),
            K_vnni.data_ptr<int8_t>(),
            scale_k.data_ptr<float>(),
            sum_k.data_ptr<int32_t>(),
            block_rowmax.data_ptr<float>(),
            running_max.data_ptr<float>(),
            block_mask.data_ptr<uint8_t>(),
            input_mask_ptr,
            (float)scale,
            (float)log_lambda, BQ, BK, D);
    };

    if (BS == 128 && STEP_KV == 128) {
        call(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
    } else if (BS == 128 && STEP_KV == 64) {
        call(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
    } else if (BS == 64 && STEP_KV == 128) {
        call(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{});
    } else if (BS == 64 && STEP_KV == 64) {
        call(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
    } else {
        TORCH_CHECK(false, "Unsupported BS/STEP_KV combo: ", BS, "/", STEP_KV,
                     ". Supported: 128/128, 128/64, 64/128, 64/64");
    }

    return std::make_tuple(block_rowmax, running_max, block_mask);
}

// ============================================================
// Torch wrapper: qk_blockmask_fp32_omp
// ============================================================
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
qk_blockmask_fp32_omp_torch(torch::Tensor Q, torch::Tensor K,
                            double log_lambda,
                            double scale,
                            c10::optional<torch::Tensor> input_mask_opt,
                            int64_t BS, int64_t STEP_KV) {
    TORCH_CHECK(Q.device().is_cpu(), "Q must be CPU");
    TORCH_CHECK(K.device().is_cpu(), "K must be CPU");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2, "Q, K must be 2D");
    Q = Q.contiguous();
    K = K.contiguous();

    int64_t BQ = Q.size(0), D = Q.size(1);
    int64_t BK = K.size(0);
    TORCH_CHECK(K.size(1) == D, "Q and K must have same D");

    int64_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    int64_t n_q_blocks = (BQ + BS - 1) / BS;

    // Validate input_mask if provided
    const uint8_t* input_mask_ptr = nullptr;
    torch::Tensor input_mask;
    if (input_mask_opt.has_value()) {
        input_mask = input_mask_opt.value().contiguous();
        TORCH_CHECK(input_mask.device().is_cpu(), "input_mask must be CPU");
        TORCH_CHECK(input_mask.dtype() == torch::kUInt8,
                     "input_mask must be uint8");
        TORCH_CHECK(input_mask.dim() == 2 &&
                     input_mask.size(0) == n_q_blocks &&
                     input_mask.size(1) == n_k_blocks,
                     "input_mask must be [n_q_blocks, n_k_blocks]");
        input_mask_ptr = input_mask.data_ptr<uint8_t>();
    }

    auto block_rowmax = torch::empty({BQ, n_k_blocks}, torch::kFloat32);
    auto running_max = torch::full({BQ}, -std::numeric_limits<float>::infinity(),
                                   torch::kFloat32);
    auto block_mask = torch::empty({n_q_blocks, n_k_blocks}, torch::kUInt8);

    auto call = [&](auto bs_tag, auto step_tag) {
        constexpr int bs_val = decltype(bs_tag)::value;
        constexpr int step_val = decltype(step_tag)::value;
        cpu_ops::qk_blockmask_fp32_omp<bs_val, step_val>(
            Q.data_ptr<float>(), K.data_ptr<float>(),
            block_rowmax.data_ptr<float>(),
            running_max.data_ptr<float>(),
            block_mask.data_ptr<uint8_t>(),
            input_mask_ptr,
            (float)scale,
            (float)log_lambda, BQ, BK, D);
    };

    if (BS == 128 && STEP_KV == 128) {
        call(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
    } else if (BS == 128 && STEP_KV == 64) {
        call(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
    } else if (BS == 64 && STEP_KV == 128) {
        call(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{});
    } else if (BS == 64 && STEP_KV == 64) {
        call(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
    } else {
        TORCH_CHECK(false, "Unsupported BS/STEP_KV combo: ", BS, "/", STEP_KV);
    }

    return std::make_tuple(block_rowmax, running_max, block_mask);
}

// ============================================================
// Module binding
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BLASST block-mask generation operators (AVX-512 VNNI + FP32)";

    m.def("pack_k_vnni", &pack_k_vnni_torch,
          "Pack FP32 K into VNNI INT8 layout with per-group quantization",
          py::arg("K"), py::arg("GS") = 4096);

    m.def("qk_blockmask_vnni_omp", &qk_blockmask_vnni_omp_torch,
          "VNNI INT8 BLASST block-mask generation (OpenMP parallel)",
          py::arg("Q"), py::arg("K_vnni"), py::arg("scale_k"),
          py::arg("sum_k"), py::arg("log_lambda"),
          py::arg("scale") = 1.0,
          py::arg("input_mask") = py::none(),
          py::arg("BS") = 128, py::arg("STEP_KV") = 128);

    m.def("qk_blockmask_fp32_omp", &qk_blockmask_fp32_omp_torch,
          "FP32 BLASST block-mask generation (OpenMP parallel)",
          py::arg("Q"), py::arg("K"), py::arg("log_lambda"),
          py::arg("scale") = 1.0,
          py::arg("input_mask") = py::none(),
          py::arg("BS") = 128, py::arg("STEP_KV") = 128);
}
