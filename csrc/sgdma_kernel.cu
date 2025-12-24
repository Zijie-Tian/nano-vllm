#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
} while (0)

/**
 * Synchronous 2D memory copy using cudaMemcpy2D.
 *
 * @param dst Destination pointer
 * @param dpitch Destination pitch (bytes) - stride between rows in destination
 * @param src Source pointer
 * @param spitch Source pitch (bytes) - stride between rows in source
 * @param width Width of data to copy per row (bytes)
 * @param height Number of rows to copy
 * @param kind Transfer direction (cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, etc.)
 */
void cuda_memcpy_2d(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind
) {
    CUDA_CHECK(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

/**
 * Asynchronous 2D memory copy using cudaMemcpy2DAsync.
 *
 * @param dst Destination pointer
 * @param dpitch Destination pitch (bytes) - stride between rows in destination
 * @param src Source pointer
 * @param spitch Source pitch (bytes) - stride between rows in source
 * @param width Width of data to copy per row (bytes)
 * @param height Number of rows to copy
 * @param kind Transfer direction
 * @param stream CUDA stream for asynchronous execution
 */
void cuda_memcpy_2d_async(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream));
}
