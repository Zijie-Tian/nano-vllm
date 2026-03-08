# Technology Stack: Nano-vLLM

## 1. Core Frameworks & Libraries
*   **PyTorch:** Primary framework for tensor operations and high-level engine logic.
*   **CUDA:** Low-level C++/CUDA implementation for optimized communication and specialized operations (e.g., sgDMA).
*   **Triton:** Used for developing high-performance GPU kernels, including block-sparse attention.
*   **Hugging Face (Transformers):** For loading model weights, configurations, and tokenizers.
*   **TVM:** Required for the **T-MAC** operator to enable efficient CPU-side coarse prediction and bit-serial interleaving packing.
*   **AVX:** Leveraged by T-MAC for hardware-accelerated, multiplication-free operations on the CPU.

## 2. System Architecture
*   **Lightweight Inference Engine:** A focused, compact implementation (~1,200 lines) residing in the `nanovllm/` directory.
*   **Optimized Communication Layer:** Low-level performance-critical logic for CPU-GPU coordination and data transfer located in the `csrc/` directory.
