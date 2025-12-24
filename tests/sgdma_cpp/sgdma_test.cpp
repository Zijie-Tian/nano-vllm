#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Configuration matching nano-vllm realistic parameters
struct Config {
    int num_layers = 32;
    int num_blocks = 10;   // Reduced from 100 to avoid huge allocation
    int block_size = 4096;
    int num_kv_heads = 8;
    int head_dim = 128;
    int dtype_size = 2;  // float16

    // Derived parameters (use size_t to avoid overflow)
    size_t features_per_block() const { return (size_t)block_size * num_kv_heads * head_dim; }
    size_t bytes_per_block() const { return features_per_block() * dtype_size; }
    int total_blocks_per_layer() const { return num_blocks; }
    size_t bytes_per_layer() const { return (size_t)num_blocks * bytes_per_block(); }
    size_t total_bytes() const { return (size_t)num_layers * bytes_per_layer(); }
};

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
};

// Initialize CPU memory with test pattern
void init_test_data(void* data, size_t bytes, int seed) {
    uint16_t* ptr = static_cast<uint16_t*>(data);
    size_t num_elements = bytes / sizeof(uint16_t);
    for (size_t i = 0; i < num_elements; i++) {
        ptr[i] = static_cast<uint16_t>((seed + i) % 65536);
    }
}

// Verify data correctness
bool verify_data(const void* data1, const void* data2, size_t bytes) {
    const uint16_t* p1 = static_cast<const uint16_t*>(data1);
    const uint16_t* p2 = static_cast<const uint16_t*>(data2);
    size_t num_elements = bytes / sizeof(uint16_t);

    for (size_t i = 0; i < num_elements; i++) {
        if (p1[i] != p2[i]) {
            std::cerr << "Mismatch at element " << i << ": "
                      << p1[i] << " != " << p2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================
// Test 1: Basic Functionality Test
// ============================================================
bool test_basic_functionality(const Config& cfg) {
    std::cout << "\n[Test 1] Basic Functionality Test" << std::endl;
    std::cout << "  Testing cudaMemcpy2D correctness with strided layout" << std::endl;

    // Allocate strided CPU memory (pinned)
    // Layout: [num_layers, num_blocks, block_features]
    size_t total_bytes = cfg.total_bytes();
    std::cout << "  Allocating " << total_bytes / 1024.0 / 1024.0 / 1024.0 << " GB pinned memory..." << std::endl;
    void* cpu_strided = nullptr;
    CUDA_CHECK(cudaMallocHost(&cpu_strided, total_bytes));
    std::cout << "  CPU strided memory allocated at: " << cpu_strided << std::endl;

    // Allocate GPU memory for one block (all layers)
    size_t gpu_block_bytes = cfg.num_layers * cfg.bytes_per_block();
    void* gpu_data = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_block_bytes));

    // Allocate CPU verify buffer
    void* cpu_verify = nullptr;
    CUDA_CHECK(cudaMallocHost(&cpu_verify, gpu_block_bytes));

    // Initialize strided CPU memory
    init_test_data(cpu_strided, total_bytes, 12345);

    // Test: Copy block_id=5 from CPU to GPU using cudaMemcpy2D
    int test_block_id = 5;
    size_t spitch = cfg.bytes_per_layer();  // Source pitch (stride between layers)
    size_t dpitch = cfg.bytes_per_block();  // Destination pitch (contiguous)
    size_t width = cfg.bytes_per_block();   // Width to copy per row
    size_t height = cfg.num_layers;         // Number of rows (layers)

    // Debug: print parameters
    std::cout << "  cudaMemcpy2D parameters:" << std::endl;
    std::cout << "    spitch: " << spitch << " bytes" << std::endl;
    std::cout << "    dpitch: " << dpitch << " bytes" << std::endl;
    std::cout << "    width: " << width << " bytes" << std::endl;
    std::cout << "    height: " << height << " rows" << std::endl;
    std::cout << "    dpitch >= width: " << (dpitch >= width ? "yes" : "no") << std::endl;
    std::cout << "    spitch >= width: " << (spitch >= width ? "yes" : "no") << std::endl;

    // Calculate source pointer (first layer, block_id)
    uint8_t* src_ptr = static_cast<uint8_t*>(cpu_strided) + test_block_id * cfg.bytes_per_block();

    // H2D transfer
    CUDA_CHECK(cudaMemcpy2D(
        gpu_data,          // dst
        dpitch,            // dpitch
        src_ptr,           // src
        spitch,            // spitch
        width,             // width
        height,            // height
        cudaMemcpyHostToDevice
    ));

    // D2H transfer back
    CUDA_CHECK(cudaMemcpy2D(
        cpu_verify,        // dst
        dpitch,            // dpitch
        gpu_data,          // src
        dpitch,            // spitch
        width,             // width
        height,            // height
        cudaMemcpyDeviceToHost
    ));

    // Verify correctness
    bool passed = true;
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        uint8_t* expected_ptr = static_cast<uint8_t*>(cpu_strided) +
                               layer * cfg.bytes_per_layer() +
                               test_block_id * cfg.bytes_per_block();
        uint8_t* actual_ptr = static_cast<uint8_t*>(cpu_verify) +
                             layer * cfg.bytes_per_block();

        if (!verify_data(expected_ptr, actual_ptr, cfg.bytes_per_block())) {
            std::cerr << "  Verification failed at layer " << layer << std::endl;
            passed = false;
            break;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(cpu_strided));
    CUDA_CHECK(cudaFreeHost(cpu_verify));
    CUDA_CHECK(cudaFree(gpu_data));

    std::cout << "  Result: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
    return passed;
}

// ============================================================
// Test 2: Performance Benchmark
// ============================================================
void test_performance_benchmark(const Config& cfg) {
    std::cout << "\n[Test 2] Performance Benchmark" << std::endl;
    std::cout << "  Configuration:" << std::endl;
    std::cout << "    num_layers: " << cfg.num_layers << std::endl;
    std::cout << "    num_blocks: " << cfg.num_blocks << std::endl;
    std::cout << "    block_size: " << cfg.block_size << std::endl;
    std::cout << "    num_kv_heads: " << cfg.num_kv_heads << std::endl;
    std::cout << "    head_dim: " << cfg.head_dim << std::endl;
    std::cout << "    dtype_size: " << cfg.dtype_size << " bytes" << std::endl;
    std::cout << "    bytes_per_block: " << cfg.bytes_per_block() / 1024.0 << " KB" << std::endl;
    std::cout << "    total transfer size: " << cfg.num_layers * cfg.bytes_per_block() / 1024.0 / 1024.0 << " MB" << std::endl;

    const int num_iterations = 100;
    const int warmup = 10;
    int test_block_id = 5;

    // Allocate memory
    size_t total_bytes = cfg.total_bytes();
    void* cpu_strided = nullptr;
    CUDA_CHECK(cudaMallocHost(&cpu_strided, total_bytes));

    void* cpu_contiguous = nullptr;
    size_t gpu_block_bytes = cfg.num_layers * cfg.bytes_per_block();
    CUDA_CHECK(cudaMallocHost(&cpu_contiguous, gpu_block_bytes));

    void* gpu_data = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_block_bytes));

    init_test_data(cpu_strided, total_bytes, 12345);
    init_test_data(cpu_contiguous, gpu_block_bytes, 12345);

    Timer timer;
    double elapsed;
    double bandwidth;

    // ========================================
    // Method A: cudaMemcpy2D with strided layout
    // ========================================
    size_t spitch = cfg.bytes_per_layer();
    size_t dpitch = cfg.bytes_per_block();
    size_t width = cfg.bytes_per_block();
    size_t height = cfg.num_layers;
    uint8_t* src_ptr = static_cast<uint8_t*>(cpu_strided) + test_block_id * cfg.bytes_per_block();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpy2D(gpu_data, dpitch, src_ptr, spitch, width, height, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemcpy2D(gpu_data, dpitch, src_ptr, spitch, width, height, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = timer.elapsed_ms();
    bandwidth = (gpu_block_bytes * num_iterations / 1e9) / (elapsed / 1000.0);

    std::cout << "\n  Method A (cudaMemcpy2D strided):" << std::endl;
    std::cout << "    Avg time: " << std::fixed << std::setprecision(3) << elapsed / num_iterations << " ms" << std::endl;
    std::cout << "    Bandwidth: " << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    double method_a_bw = bandwidth;

    // ========================================
    // Method B: cudaMemcpy with contiguous layout (baseline)
    // ========================================
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpy(gpu_data, cpu_contiguous, gpu_block_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemcpy(gpu_data, cpu_contiguous, gpu_block_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = timer.elapsed_ms();
    bandwidth = (gpu_block_bytes * num_iterations / 1e9) / (elapsed / 1000.0);

    std::cout << "\n  Method B (cudaMemcpy contiguous):" << std::endl;
    std::cout << "    Avg time: " << std::fixed << std::setprecision(3) << elapsed / num_iterations << " ms" << std::endl;
    std::cout << "    Bandwidth: " << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    double method_b_bw = bandwidth;

    // ========================================
    // Method C: Layer-by-layer copy (simulate PyTorch non-contiguous)
    // ========================================
    // Warmup
    for (int i = 0; i < warmup; i++) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            uint8_t* src_layer = static_cast<uint8_t*>(cpu_strided) +
                                 layer * cfg.bytes_per_layer() +
                                 test_block_id * cfg.bytes_per_block();
            uint8_t* dst_layer = static_cast<uint8_t*>(gpu_data) + layer * cfg.bytes_per_block();
            CUDA_CHECK(cudaMemcpy(dst_layer, src_layer, cfg.bytes_per_block(), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    timer.start();
    for (int i = 0; i < num_iterations; i++) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            uint8_t* src_layer = static_cast<uint8_t*>(cpu_strided) +
                                 layer * cfg.bytes_per_layer() +
                                 test_block_id * cfg.bytes_per_block();
            uint8_t* dst_layer = static_cast<uint8_t*>(gpu_data) + layer * cfg.bytes_per_block();
            CUDA_CHECK(cudaMemcpy(dst_layer, src_layer, cfg.bytes_per_block(), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = timer.elapsed_ms();
    bandwidth = (gpu_block_bytes * num_iterations / 1e9) / (elapsed / 1000.0);

    std::cout << "\n  Method C (layer-by-layer copy):" << std::endl;
    std::cout << "    Avg time: " << std::fixed << std::setprecision(3) << elapsed / num_iterations << " ms" << std::endl;
    std::cout << "    Bandwidth: " << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    double method_c_bw = bandwidth;

    // Summary
    std::cout << "\n  ========================================" << std::endl;
    std::cout << "  Performance Summary:" << std::endl;
    std::cout << "    Method A vs Method B: " << std::setprecision(2) << (method_a_bw / method_b_bw * 100) << "%" << std::endl;
    std::cout << "    Method A vs Method C: " << std::setprecision(2) << (method_a_bw / method_c_bw) << "x speedup" << std::endl;
    std::cout << "  ========================================" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFreeHost(cpu_strided));
    CUDA_CHECK(cudaFreeHost(cpu_contiguous));
    CUDA_CHECK(cudaFree(gpu_data));
}

int main() {
    std::cout << "=== cudaMemcpy2D Test ===" << std::endl;

    // Print CUDA device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " <<
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;

    Config cfg;

    // Run tests
    bool test1_passed = test_basic_functionality(cfg);
    test_performance_benchmark(cfg);

    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "All tests " << (test1_passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

    return test1_passed ? 0 : 1;
}
