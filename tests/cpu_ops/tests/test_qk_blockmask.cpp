/**
 * test_qk_blockmask.cpp - Correctness + Benchmark for BLASST block-mask
 */

#include "cpu_ops/qk_blockmask.h"
#include "cpu_ops/qk_rowmax.h"
#include <benchmark/benchmark.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void fill_random(float* data, size_t n, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}

// ============================================================
// Correctness: VNNI vs FP32 reference
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
static void verify_blockmask(size_t BQ, size_t BK, size_t D, float lambda) {
    std::vector<float> Q(BQ * D), K(BK * D);
    fill_random(Q.data(), Q.size(), 42);
    fill_random(K.data(), K.size(), 43);

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;
    float log_lambda = std::log(lambda);

    // FP32 reference
    std::vector<float> ref_brm(BQ * n_k_blocks);
    std::vector<float> ref_rmax(BQ, -std::numeric_limits<float>::infinity());
    std::vector<uint8_t> ref_mask(n_q_blocks * n_k_blocks);
    cpu_ops::qk_blockmask_fp32<BS, STEP_KV>(
        Q.data(), K.data(), ref_brm.data(), ref_rmax.data(),
        ref_mask.data(), log_lambda, BQ, BK, D);

    // VNNI
    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);

    std::vector<float> vnni_brm(BQ * n_k_blocks);
    std::vector<float> vnni_rmax(BQ, -std::numeric_limits<float>::infinity());
    std::vector<uint8_t> vnni_mask(n_q_blocks * n_k_blocks);
    cpu_ops::qk_blockmask_vnni<MR, NR, GS, BS, STEP_KV>(
        Q.data(), Kv, sk.data(), sumk.data(),
        vnni_brm.data(), vnni_rmax.data(), vnni_mask.data(),
        log_lambda, BQ, BK, D);

    float max_rel = 0;
    for (size_t idx = 0; idx < BQ * n_k_blocks; ++idx) {
        float rel = (std::fabs(ref_brm[idx]) > 1e-6f)
            ? std::fabs(vnni_brm[idx] - ref_brm[idx]) / std::fabs(ref_brm[idx])
            : std::fabs(vnni_brm[idx] - ref_brm[idx]);
        max_rel = std::max(max_rel, rel);
    }
    size_t mask_diffs = 0;
    for (size_t idx = 0; idx < n_q_blocks * n_k_blocks; ++idx)
        if (ref_mask[idx] != vnni_mask[idx]) ++mask_diffs;

    printf("  rowmax_err=%.3f%% mask_diffs=%zu/%zu\n",
           max_rel * 100, mask_diffs, n_q_blocks * n_k_blocks);
    assert(max_rel < 0.05f);
    cpu_ops::free_aligned(Kv);
}

// ============================================================
// Threshold sweep
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
static void verify_threshold_sweep(size_t BQ, size_t BK, size_t D) {
    std::vector<float> Q(BQ * D), K(BK * D);
    fill_random(Q.data(), Q.size(), 42);
    fill_random(K.data(), K.size(), 43);

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;
    size_t mask_size = n_q_blocks * n_k_blocks;

    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);

    float lambdas[] = { 1e-10f, 1e-5f, 1e-3f, 0.1f, 0.5f, 0.9f };
    printf("  λ sweep (BQ=%zu, BK=%zu, BS=%d, STEP_KV=%d):\n", BQ, BK, BS, STEP_KV);
    for (float lambda : lambdas) {
        std::vector<float> brm(BQ * n_k_blocks);
        std::vector<float> rmax(BQ, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> mask(mask_size);
        cpu_ops::qk_blockmask_vnni<MR, NR, GS, BS, STEP_KV>(
            Q.data(), Kv, sk.data(), sumk.data(),
            brm.data(), rmax.data(), mask.data(),
            std::log(lambda), BQ, BK, D);
        size_t keeps = 0;
        for (size_t j = 0; j < mask_size; ++j) keeps += mask[j];
        printf("    λ=%.1e → keep=%zu/%zu (%.1f%%)\n",
               lambda, keeps, mask_size, 100.0 * keeps / mask_size);
    }

    // λ=1e-10 must keep everything
    {
        std::vector<float> brm(BQ * n_k_blocks);
        std::vector<float> rmax(BQ, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> mask(mask_size);
        cpu_ops::qk_blockmask_vnni<MR, NR, GS, BS, STEP_KV>(
            Q.data(), Kv, sk.data(), sumk.data(),
            brm.data(), rmax.data(), mask.data(),
            std::log(1e-10f), BQ, BK, D);
        size_t keeps = 0;
        for (size_t j = 0; j < mask_size; ++j) keeps += mask[j];
        assert(keeps == mask_size && "λ=1e-10 should keep all");
    }
    cpu_ops::free_aligned(Kv);
}

static void verify_correctness() {
    printf("[TEST] blockmask <8,2,4096,128,128> D=16:\n");
    verify_blockmask<8, 2, 4096, 128, 128>(256, 4096, 16, 0.1f);
    printf("[OK]\n");

    printf("[TEST] blockmask <8,2,4096,128,64> D=16:\n");
    verify_blockmask<8, 2, 4096, 128, 64>(256, 4096, 16, 0.1f);
    printf("[OK]\n");

    printf("[TEST] blockmask <8,2,4096,64,128> D=8:\n");
    verify_blockmask<8, 2, 4096, 64, 128>(256, 4096, 8, 0.1f);
    printf("[OK]\n");

    verify_threshold_sweep<8, 2, 4096, 128, 128>(256, 4096, 16);
    printf("[OK] threshold sweep.\n");

    // FP32 OMP vs naive correctness
    printf("[TEST] FP32 OMP vs naive <128,128> D=16:\n");
    {
        size_t BQ = 256, BK = 4096, D = 16;
        constexpr int BS = 128, STEP_KV = 128;
        size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
        size_t n_q_blocks = (BQ + BS - 1) / BS;
        std::vector<float> Q(BQ * D), K(BK * D);
        fill_random(Q.data(), Q.size(), 42);
        fill_random(K.data(), K.size(), 43);
        float log_lambda = std::log(0.1f);

        std::vector<float> ref_brm(BQ * n_k_blocks), omp_brm(BQ * n_k_blocks);
        std::vector<float> ref_rmax(BQ, -std::numeric_limits<float>::infinity());
        std::vector<float> omp_rmax(BQ, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> ref_mask(n_q_blocks * n_k_blocks);
        std::vector<uint8_t> omp_mask(n_q_blocks * n_k_blocks);

        cpu_ops::qk_blockmask_fp32<BS, STEP_KV>(
            Q.data(), K.data(), ref_brm.data(), ref_rmax.data(),
            ref_mask.data(), log_lambda, BQ, BK, D);
        cpu_ops::qk_blockmask_fp32_omp<BS, STEP_KV>(
            Q.data(), K.data(), omp_brm.data(), omp_rmax.data(),
            omp_mask.data(), log_lambda, BQ, BK, D);

        size_t mask_diffs = 0;
        float max_rel = 0;
        for (size_t idx = 0; idx < BQ * n_k_blocks; ++idx) {
            float rel = (std::fabs(ref_brm[idx]) > 1e-6f)
                ? std::fabs(omp_brm[idx] - ref_brm[idx]) / std::fabs(ref_brm[idx])
                : std::fabs(omp_brm[idx] - ref_brm[idx]);
            max_rel = std::max(max_rel, rel);
        }
        for (size_t idx = 0; idx < n_q_blocks * n_k_blocks; ++idx)
            if (ref_mask[idx] != omp_mask[idx]) ++mask_diffs;
        printf("  rowmax_err=%.6f%% mask_diffs=%zu/%zu\n",
               max_rel * 100, mask_diffs, n_q_blocks * n_k_blocks);
        assert(max_rel < 1e-5f && "FP32 OMP should match naive exactly");
        assert(mask_diffs == 0 && "FP32 OMP masks must match naive");
    }
    printf("[OK]\n");
}

// ============================================================
// Benchmarks
// ============================================================
template <int MR, int NR, int GS, int BS, int STEP_KV>
static void BM_blockmask_omp(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;
    size_t ng = cpu_ops::num_quant_groups(BK, GS);
    int8_t* Kv = cpu_ops::alloc_aligned_i8(cpu_ops::pack_k_vnni_buffer_size(BK, D));
    std::vector<float> sk(ng); std::vector<int32_t> sumk(BK + 16, 0);
    cpu_ops::pack_k_vnni(K.data(), Kv, sk.data(), sumk.data(), BK, D, GS);

    std::vector<float> brm(BQ * n_k_blocks);
    std::vector<uint8_t> mask(n_q_blocks * n_k_blocks);

    for (auto _ : state) {
        std::vector<float> rmax(BQ, -std::numeric_limits<float>::infinity());
        cpu_ops::qk_blockmask_vnni_omp<MR, NR, GS, BS, STEP_KV>(
            Q.data(), Kv, sk.data(), sumk.data(),
            brm.data(), rmax.data(), mask.data(),
            std::log(0.1f), BQ, BK, D);
        benchmark::DoNotOptimize(mask.data());
    }
    cpu_ops::free_aligned(Kv);
    state.counters["GOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

template <int BS, int STEP_KV>
static void BM_blockmask_fp32_omp(benchmark::State& state) {
    size_t BQ = state.range(0), BK = state.range(1), D = state.range(2);
    std::vector<float> Q(BQ * D), K(BK * D);
    fill_random(Q.data(), Q.size(), 42); fill_random(K.data(), K.size(), 43);

    size_t n_k_blocks = (BK + STEP_KV - 1) / STEP_KV;
    size_t n_q_blocks = (BQ + BS - 1) / BS;
    std::vector<float> brm(BQ * n_k_blocks);
    std::vector<uint8_t> mask(n_q_blocks * n_k_blocks);

    for (auto _ : state) {
        std::vector<float> rmax(BQ, -std::numeric_limits<float>::infinity());
        cpu_ops::qk_blockmask_fp32_omp<BS, STEP_KV>(
            Q.data(), K.data(),
            brm.data(), rmax.data(), mask.data(),
            std::log(0.1f), BQ, BK, D);
        benchmark::DoNotOptimize(mask.data());
    }
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * BQ * BK * D / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_blockmask_omp<8, 2, 4096, 128, 128>)
    ->Args({4096, 32768, 16})->Args({4096, 32768, 8})->Args({4096, 32768, 32})
    ->Args({4096, 131072, 16})->Args({4096, 131072, 8})->Args({4096, 131072, 32})
    ->Args({4096, 1048576, 16})->Args({4096, 1048576, 8})->Args({4096, 1048576, 32})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

BENCHMARK(BM_blockmask_fp32_omp<128, 128>)
    ->Args({4096, 32768, 16})->Args({4096, 32768, 8})->Args({4096, 32768, 32})
    ->Unit(benchmark::kMillisecond)->MinWarmUpTime(1.0);

int main(int argc, char** argv) {
    verify_correctness();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

