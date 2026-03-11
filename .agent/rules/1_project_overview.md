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
- `docs/architecture_guide.md`: Core components, CPU offload design, ring buffer.
- `docs/sparse_policy_architecture.md`: SparsePolicy abstraction and pipeline modes.
- `docs/sparse_attention_guide.md`: Block sparse attention methods (XAttention, MInference, etc.).
- `docs/sparse_attention_blasst.md`: BLASST sparse attention: dynamic pruning, online softmax thresholding.
- `docs/blasst_performance_analysis.md`: Performance report for BLASST: λ vs density, accuracy stability.
- `docs/blasst_mask_visualization_guide.md`: Step-by-step guide to export and plot BLASST attention masks.
- `docs/xattention_algorithm_guide.md`: XAttention algorithm details & Triton kernels.
- `docs/debugging_guide.md`: PyTorch hooks, tensor comparison, memory profiling.
- `docs/optimization_guide.md`: Performance optimizations (sgDMA, Triton merge).
- `docs/test_ruler_usage_guide.md`: Comprehensive guide for test_ruler.py.
- `docs/known_issues.md`: Documented bugs and resolution history.
- `docs/trtllm_skip_softmax_implementation_details.md`: Deep dive into Skip Softmax (BLASST) in TensorRT-LLM: math, kernels, and ModelOpt.
- `docs/sparse_policy_offload_control_guide.md`: Guide for controlling KV cache offloading within custom SparsePolicies.
- `docs/tvm_knowledge_base.md`: TVM troubleshooting and optimization: fallback warnings, x86 stability, etc.
- `docs/kcache_quantization_guide.md`: 2-bit per-token K-cache quantization for T-MAC.
- `docs/tmac_qgemm_deep_dive.md`: Technical deep dive into T-MAC QGEMM: algorithm, layout, and Phase 1 optimizations.
- `docs/tmac_tvm_qgemm_migration_guide.md`: Migration guide for TMAC TVM QGEMM.
