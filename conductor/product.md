# Initial Concept

Infinite context LLM inference on a single 24GB GPU through CPU-GPU Heterogeneous Offloading and Dynamic Sparse Attention.

# Product Definition: Nano-vLLM

## 1. Product Vision: Infinite Context on Single RTX 3090
The ultimate goal of Nano-vLLM is to achieve infinite context LLM inference on a single 24GB GPU (e.g., RTX 3090/4090) through **CPU-GPU Heterogeneous Offloading** and **Dynamic Sparse Attention**. This project is a lightweight implementation (~1,200 lines) dedicated to fast offline LLM inference for models like Qwen2/3, Llama-3, and GLM-4.

## 2. System Pipeline Philosophy
The architecture is built around a specialized pipeline designed to overcome memory constraints:
*   **GPU Fused Generation/Prediction Proxy:** Generates high-precision FP16 KV cache for offloading while performing SRAM-level quantization and interleaving packing.
*   **CPU T-MAC Coarse Predictor:** Uses AVX instructions to build LUTs and perform multiplication-free mpGEMM on compressed 2-bit KV cache to generate a globally shared coarse-grained mask.
*   **I/O Scheduler & Delta Transfer:** Compacts scattered FP16 KV blocks in pinned memory based on the CPU mask and initiates a single asynchronous bulk DMA transfer using a diff against the GPU cache.
*   **Block-Sparse Attention Kernel:** A Triton/CUDA kernel (BLASST) that performs chunked prefill and dynamic pruning (skipping Softmax and PV operations) using LSE thresholds.

## 3. Core Objectives
*   **Infinite Context:** Enabling 128k, 256k, and even 1m+ token inference on a single consumer GPU.
*   **Minimal Overhead:** Zero-overhead fine-grained computation through efficient CPU-GPU coordination.
*   **High Performance:** Maintaining high-speed inference through specialized kernels and an optimized I/O scheduler.
*   **Architectural Purity:** A focused, lightweight implementation (~1,200 lines) that prioritizes this specialized offloading and sparse attention pipeline over generic features.

## 4. Hardware Mandates
*   **RTX 3090/4090 (24GB):** Must use `--enable-offload` for 7B+ models or long contexts to avoid OOM.
*   **A100 (40/80GB):** Supports both offload and GPU-only modes.
*   **Long-Context Validation:** Always use `--enable-offload --input-len 32768` as the baseline for validation.
