---
description: Nano-vLLM Project Overview and Architectural Vision
alwaysApply: false
---
# Nano-vLLM Project Overview

Nano-vLLM is a lightweight (~1,200 lines) implementation for fast offline LLM inference. It supports models like Qwen2/3, Llama-3, and GLM-4, featuring a specialized CPU offload system for long-context inference on consumer GPUs (e.g., RTX 3090/4090).

## Far-Term Architectural Vision (Infinite Context on Single RTX 3090)
The ultimate goal of this project is to achieve infinite context LLM inference on a single 24GB GPU through **CPU-GPU Heterogeneous Offloading** and **Dynamic Sparse Attention**.
The system pipeline philosophy is: **GPU Fused Generation/Prediction Proxy -> CPU T-MAC Fast Coarse Filtering -> Shared Masking -> PCIe On-Demand Delta Transfer -> GPU BLASST Zero-Overhead Fine-Grained Computation**.

## Architecture Modules
*   **Module A (GPU Fused Epilogue Kernel)**: Generates high-precision FP16 KV cache for offloading while simultaneously performing SRAM-level Warp mean-pooling, 2-bit quantization, and T-MAC bit-serial interleaving packing.
*   **Module B (CPU T-MAC Coarse Predictor)**: Uses AVX instructions to build LUTs and perform multiplication-free mpGEMM on the compressed 2-bit KV cache, generating a globally shared coarse-grained mask.
*   **Module C (I/O Scheduler & Delta Transfer)**: Compacts scattered FP16 KV blocks in pinned memory based on the CPU mask and initiates a single asynchronous bulk DMA transfer using a diff against the GPU cache.
*   **Module D (Block-Sparse Attention Kernel)**: A Triton/CUDA kernel based on FlashInfer and BLASST that performs chunked prefill and dynamic pruning (skipping Softmax and PV operations) using LSE thresholds.

## Configuration Reference
*   `kvcache_block_size`: 1024 (Tokens per block, 4096 supported).
*   `max_num_batched_tokens`: 16384 (Set = max_model_len for long context).
*   `gpu_memory_utilization`: 0.9 (Target GPU memory fraction).
*   `enable_cpu_offload`: False (Enable for long context - required on 3090/4090).
*   `enforce_eager`: False (Set True to disable CUDA graphs, useful for debugging).

## Documentation Index
- Core architecture: `docs/architecture_guide.md`, `docs/sparse_policy_architecture.md`
- BLASST/Block Sparse: `docs/sparse_attention_guide.md`, `docs/sparse_attention_blasst.md`, `docs/blasst_performance_analysis.md`, `docs/blasst_mask_visualization_guide.md`
- XAttention: `docs/xattention_algorithm_guide.md`
- Debugging/Perf: `docs/debugging_guide.md`, `docs/optimization_guide.md`, `docs/test_ruler_usage_guide.md`, `docs/known_issues.md`
- TVM/Quantization: `docs/tvm_knowledge_base.md`, `docs/kcache_quantization_guide.md`, `docs/tmac_qgemm_deep_dive.md`
