#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Forward declarations of CUDA functions from sgdma_kernel.cu
void cuda_memcpy_2d(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind
);

void cuda_memcpy_2d_async(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream
);

// Helper function to parse memcpy kind string to enum
cudaMemcpyKind parse_memcpy_kind(const std::string& kind_str) {
    if (kind_str == "h2d") {
        return cudaMemcpyHostToDevice;
    } else if (kind_str == "d2h") {
        return cudaMemcpyDeviceToHost;
    } else if (kind_str == "d2d") {
        return cudaMemcpyDeviceToDevice;
    } else if (kind_str == "h2h") {
        return cudaMemcpyHostToHost;
    } else {
        throw std::invalid_argument("Invalid memcpy kind. Must be one of: h2d, d2h, d2d, h2h");
    }
}

/**
 * PyTorch wrapper for cudaMemcpy2D (synchronous).
 *
 * @param dst Destination tensor
 * @param src Source tensor
 * @param dpitch Destination pitch in bytes
 * @param spitch Source pitch in bytes
 * @param width Width to copy per row in bytes
 * @param height Number of rows
 * @param kind Transfer direction ("h2d", "d2h", "d2d", "h2h")
 */
void memcpy_2d_torch(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t dpitch,
    int64_t spitch,
    int64_t width,
    int64_t height,
    const std::string& kind
) {
    // Parse kind string
    cudaMemcpyKind kind_enum = parse_memcpy_kind(kind);

    // Basic validation
    if (dpitch < width) {
        throw std::invalid_argument("dpitch must be >= width");
    }
    if (spitch < width) {
        throw std::invalid_argument("spitch must be >= width");
    }

    // Validate tensor devices match the memcpy kind
    bool src_is_cuda = src.device().is_cuda();
    bool dst_is_cuda = dst.device().is_cuda();

    if (kind == "h2d") {
        if (src_is_cuda) {
            throw std::invalid_argument("Source must be on CPU for h2d transfer");
        }
        if (!dst_is_cuda) {
            throw std::invalid_argument("Destination must be on CUDA for h2d transfer");
        }
    } else if (kind == "d2h") {
        if (!src_is_cuda) {
            throw std::invalid_argument("Source must be on CUDA for d2h transfer");
        }
        if (dst_is_cuda) {
            throw std::invalid_argument("Destination must be on CPU for d2h transfer");
        }
    } else if (kind == "d2d") {
        if (!src_is_cuda || !dst_is_cuda) {
            throw std::invalid_argument("Both source and destination must be on CUDA for d2d transfer");
        }
    }

    // Get raw pointers
    void* dst_ptr = dst.data_ptr();
    const void* src_ptr = src.data_ptr();

    // Call CUDA function
    cuda_memcpy_2d(
        dst_ptr,
        static_cast<size_t>(dpitch),
        src_ptr,
        static_cast<size_t>(spitch),
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        kind_enum
    );
}

/**
 * PyTorch wrapper for cudaMemcpy2DAsync (asynchronous).
 *
 * @param dst Destination tensor
 * @param src Source tensor
 * @param dpitch Destination pitch in bytes
 * @param spitch Source pitch in bytes
 * @param width Width to copy per row in bytes
 * @param height Number of rows
 * @param kind Transfer direction ("h2d", "d2h", "d2d", "h2h")
 * @param stream_ptr CUDA stream pointer as int64_t (from torch.cuda.Stream.cuda_stream)
 */
void memcpy_2d_async_torch(
    torch::Tensor dst,
    torch::Tensor src,
    int64_t dpitch,
    int64_t spitch,
    int64_t width,
    int64_t height,
    const std::string& kind,
    int64_t stream_ptr
) {
    // Parse kind string
    cudaMemcpyKind kind_enum = parse_memcpy_kind(kind);

    // Basic validation (same as sync version)
    if (dpitch < width) {
        throw std::invalid_argument("dpitch must be >= width");
    }
    if (spitch < width) {
        throw std::invalid_argument("spitch must be >= width");
    }

    // Validate tensor devices
    bool src_is_cuda = src.device().is_cuda();
    bool dst_is_cuda = dst.device().is_cuda();

    if (kind == "h2d") {
        if (src_is_cuda) {
            throw std::invalid_argument("Source must be on CPU for h2d transfer");
        }
        if (!dst_is_cuda) {
            throw std::invalid_argument("Destination must be on CUDA for h2d transfer");
        }
    } else if (kind == "d2h") {
        if (!src_is_cuda) {
            throw std::invalid_argument("Source must be on CUDA for d2h transfer");
        }
        if (dst_is_cuda) {
            throw std::invalid_argument("Destination must be on CPU for d2h transfer");
        }
    } else if (kind == "d2d") {
        if (!src_is_cuda || !dst_is_cuda) {
            throw std::invalid_argument("Both source and destination must be on CUDA for d2d transfer");
        }
    }

    // Get raw pointers
    void* dst_ptr = dst.data_ptr();
    const void* src_ptr = src.data_ptr();

    // Cast stream pointer
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Call CUDA function
    cuda_memcpy_2d_async(
        dst_ptr,
        static_cast<size_t>(dpitch),
        src_ptr,
        static_cast<size_t>(spitch),
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        kind_enum,
        stream
    );
}

// Python module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA sgDMA (cudaMemcpy2D) extension for PyTorch";

    m.def("memcpy_2d", &memcpy_2d_torch,
        "Synchronous 2D memory copy using cudaMemcpy2D",
        py::arg("dst"),
        py::arg("src"),
        py::arg("dpitch"),
        py::arg("spitch"),
        py::arg("width"),
        py::arg("height"),
        py::arg("kind")
    );

    m.def("memcpy_2d_async", &memcpy_2d_async_torch,
        "Asynchronous 2D memory copy using cudaMemcpy2DAsync",
        py::arg("dst"),
        py::arg("src"),
        py::arg("dpitch"),
        py::arg("spitch"),
        py::arg("width"),
        py::arg("height"),
        py::arg("kind"),
        py::arg("stream_ptr")
    );
}
