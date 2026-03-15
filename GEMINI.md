# GEMINI.md

This file provides foundational mandates and guidance for Gemini CLI when working with the Nano-vLLM repository.

## Project Overview

Nano-vLLM is a lightweight (~1,200 lines) implementation for fast offline LLM inference. It supports models like Qwen2/3, Llama-3, and GLM-4, featuring a specialized CPU offload system for long-context inference on consumer GPUs (e.g., RTX 3090/4090).

**Far-Term Architectural Vision (Infinite Context on Single RTX 3090)**
The ultimate goal of this project is to achieve infinite context LLM inference on a single 24GB GPU through **CPU-GPU Heterogeneous Offloading** and **Dynamic Sparse Attention**.
The system pipeline philosophy is: **GPU Fused Generation/Prediction Proxy -> CPU T-MAC Fast Coarse Filtering & Shared Masking -> PCIe On-Demand Delta Transfer -> GPU BLASST Zero-Overhead Fine-Grained Computation**.

The architecture consists of 4 core modules:
*   **Module A (GPU Fused Epilogue Kernel)**: Generates high-precision FP16 KV cache for offloading while simultaneously performing SRAM-level Warp mean-pooling, 2-bit quantization, and T-MAC bit-serial interleaving packing.
*   **Module B (CPU T-MAC Coarse Predictor)**: Uses AVX instructions to build LUTs and perform multiplication-free mpGEMM on the compressed 2-bit KV cache, generating a globally shared coarse-grained mask.
*   **Module C (I/O Scheduler & Delta Transfer)**: Compacts scattered FP16 KV blocks in pinned memory based on the CPU mask and initiates a single asynchronous bulk DMA transfer using a diff against the GPU cache.
*   **Module D (Block-Sparse Attention Kernel)**: A Triton/CUDA kernel based on FlashInfer and BLASST that performs chunked prefill and dynamic pruning (skipping Softmax and PV operations) using LSE thresholds.

---

## 1. Engineering Standards & Mandates

### 1.1 Multi-GPU Debugging & Resource Allocation
*   **Conservative Allocation**: For long-running validations (>20 mins or >50 samples), use only 1-2 GPUs.
*   **Parallel Exploration**: Reserve remaining GPUs (at least 50% if ≥4 GPUs are available) for parallel hypothesis testing and fast iteration (≤10 samples).
*   **Single-Task Validation**: During debugging, focus on ONE representative task first. Do not run the full benchmark suite until the fix is verified.
*   **Non-Blocking Monitoring**: Always run long tests in the background (`is_background: true`) and continue analysis or exploratory tests on other GPUs.

### 1.2 GPU Testing & Hardware Rules
*   **RTX 3090 / 4090 (24GB)**: **MUST** use `--enable-offload`. GPU-only mode will OOM for 7B+ models.
*   **A100 (40/80GB)**: Both offload and GPU-only modes are acceptable.
*   **Needle Test Requirement**: Always use `--enable-offload --input-len 32768`. 8K is insufficient for validation.
*   **GPU Mutex**: Before running benchmarks (`bench*.py`), wait for exclusive GPU access if other processes are using the compute apps.

### 1.3 Sparse Policy Implementation
*   **No `None` Policy**: `sparse_policy` must never be `None`. Default to `FullAttentionPolicy` if unspecified.
*   **OffloadEngine Communication**: All CPU-GPU data transfers in policies **MUST** go through `OffloadEngine` (e.g., `load_to_slot_layer`, `wait_slot_layer`) to ensure stream synchronization and pipeline optimization. **Direct `.to("cuda")` or `.copy_()` is prohibited in compute methods.**
*   **Interface Compliance**: Policies must declare `supports_prefill`/`supports_decode` and implement `select_blocks()`, `compute_chunked_prefill()`, and `compute_chunked_decode()`.

### 1.4 Low-Level Optimization (Triton/CUDA)
*   **Design First**: Always create a design specification before implementing Triton kernels, including algorithm overview, interface, and numerical constraints.
*   **Reference Implementation**: Always provide a naive PyTorch reference for correctness validation.
*   **Validation**: Kernels must be tested for correctness (matching reference), numerical stability (no NaN/Inf), and performance (speedup vs baseline).

### 1.5 Testing & Profiling Standards
*   **Test Style**: Minimal prints, use structured data (e.g., all ones) for easy manual verification, use `assert` for validation, and only print `test_xxx: PASSED` at the end.
*   **Profiling (nsys)**: **MUST** use `scripts/profile_offload.sh`. Do NOT run `nsys` directly.
*   **Benchmarking**: Before running `bench*.py`, ensure exclusive GPU access by checking for other compute-intensive PIDs.

---

## 2. Operational Procedures

### 2.1 GPU Selection Protocol (MANDATORY)
1.  **Check Context**: Look for explicit GPU IDs in the user's prompt or history.
2.  **Ask if Unclear**: If no GPU is specified, **STOP and ASK** the user which GPU(s) to use.
3.  **Prefix Commands**: Always use `CUDA_VISIBLE_DEVICES=X` for most GPU-related commands.
4.  **Script Exception**: For `scripts/profile_offload.sh`, use the `--gpu X` argument instead of `CUDA_VISIBLE_DEVICES`.

### 2.2 Environment & Testing
*   **Environment Setup (MANDATORY)**: Before executing **any** code or scripts in this repository, you **MUST** configure the environment by sourcing `build/nano-vllm-envs.sh`. This script configures `PYTHONPATH=$(pwd):$PYTHONPATH` and sets up TVM paths.
    *   **Pre-execution Check**: Always check if `build/nano-vllm-envs.sh` exists.
    *   **TVM Configuration**: If the script does *not* exist, you must configure TVM first by running: `python3 scripts/setup_tvm.py`. Wait for the build to complete, then source the script: `source build/nano-vllm-envs.sh`.
*   **test_ruler.py**: Read `docs/test_ruler_usage_guide.md` before running. Do not use `--help`. Match `data-dir` with appropriate `max-model-len`.
*   **Documentation Indexing**: Whenever a new document is added to the `docs/` directory, its path and purpose **MUST** be immediately indexed in both `GEMINI.md` and `CLAUDE.md`.
*   **Planning Files**: Use `findings.md`, `task_plan.md`, and `progress.md` for complex tasks. These are excluded from git. **At the beginning of every new task, you MUST automatically delete any existing `task_plan.md`, `findings.md`, and `progress.md` files to ensure a fresh state.**

### 2.3 Monitoring
*   **GPU Monitoring**: For profiling or OOM debugging, run monitoring commands in the background. Prefer `nvidia-smi` queries or specialized profiling tools (nsys) directed to background output files.

---

## 3. Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture_guide.md`](docs/architecture_guide.md) | Core components, CPU offload design, ring buffer. |
| [`docs/sparse_policy_architecture.md`](docs/sparse_policy_architecture.md) | SparsePolicy abstraction and pipeline modes. |
| [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md) | Block sparse attention methods (XAttention, MInference, etc.). |
| [`docs/sparse_attention_blasst.md`](docs/sparse_attention_blasst.md) | BLASST sparse attention: dynamic pruning, online softmax thresholding. |
| [`docs/blasst_performance_analysis.md`](docs/blasst_performance_analysis.md) | Performance report for BLASST: λ vs density, accuracy stability. |
| [`docs/blasst_mask_visualization_guide.md`](docs/blasst_mask_visualization_guide.md) | Step-by-step guide to export and plot BLASST attention masks. |
| [`docs/xattention_algorithm_guide.md`](docs/xattention_algorithm_guide.md) | XAttention algorithm details & Triton kernels. |
| [`docs/debugging_guide.md`](docs/debugging_guide.md) | PyTorch hooks, tensor comparison, memory profiling. |
| [`docs/optimization_guide.md`](docs/optimization_guide.md) | Performance optimizations (sgDMA, Triton merge). |
| [`docs/test_ruler_usage_guide.md`](docs/test_ruler_usage_guide.md) | Comprehensive guide for `test_ruler.py`. |
| [`docs/known_issues.md`](docs/known_issues.md) | Documented bugs and resolution history. |
| [`docs/trtllm_skip_softmax_implementation_details.md`](docs/trtllm_skip_softmax_implementation_details.md) | Deep dive into Skip Softmax (BLASST) in TensorRT-LLM: math, kernels, and ModelOpt. |
| [`docs/sparse_policy_offload_control_guide.md`](docs/sparse_policy_offload_control_guide.md) | Guide for controlling KV cache offloading within custom SparsePolicies. |
| [`docs/tvm_knowledge_base.md`](docs/tvm_knowledge_base.md) | TVM troubleshooting and optimization: fallback warnings, x86 stability, etc. |
| [`docs/kcache_quantization_guide.md`](docs/kcache_quantization_guide.md) | 2-bit per-token K-cache quantization for T-MAC. |
| [`docs/tmac_qgemm_deep_dive.md`](docs/tmac_qgemm_deep_dive.md) | Technical deep dive into T-MAC QGEMM: algorithm, layout, and Phase 1 optimizations. |
| [`docs/tmac_tvm_qgemm_migration_guide.md`](docs/tmac_tvm_qgemm_migration_guide.md) | Migration guide: T-MAC QGeMM TVM integration, architectural placement, and tests. |
| [`docs/blasst_bugfix_mglobal_causal.md`](docs/blasst_bugfix_mglobal_causal.md) | BLASST bugfix report: m_global/LSE decoupling & element-wise causal masking. |
| [`docs/blasst_perhead_density_analysis.md`](docs/blasst_perhead_density_analysis.md) | Per-head KV density analysis: GQA IO implications for sparse attention offload. |
| [`docs/blasst_density_collection_guide.md`](docs/blasst_density_collection_guide.md) | BLASST density 数据采集全流程指南。步骤：`test_ruler.py --sparse-policy BLASST` 采集日志 → `scripts/export_blasst_density_csv.py` 解析并导出 per-layer CSV（rows=chunk, cols=head）→ `scripts/analyze_blasst_density.py` 生成汇总分析。数据存放于 `results/density/`。 |
| [`docs/compass_tmac_l1_batching.md`](docs/compass_tmac_l1_batching.md) | COMPASS TMAC L1 批量化优化全记录：32 线程 M 维度并行、pre-compute per-head 优化、端到端 benchmark（4.7s/层）、Python 封装开销分析（`_preprocess_q_per_head` 占 81%）以及下一步优化路径。 |
| [`docs/compass_torch_gemm_migration.md`](docs/compass_torch_gemm_migration.md) | COMPASS Torch GEMM estimation: TMAC→Torch migration, BLASST-consistent per-token scoring algorithm, BMM optimization, and benchmark results. |
| [`docs/compass_analysis.md`](docs/compass_analysis.md) | COMPASS pooled top-p analysis: GPU Q pooling optimization, CPU profiling breakdown, BLASST λ sweep, top_p sweep, two-level sparsity constraint, per-head IO directions. |
| [`docs/compass_subblock_compacted_transfer.md`](docs/compass_subblock_compacted_transfer.md) | COMPASS sub-block compacted transfer: gather+bulk H2D pipeline, top-p Q-aggregation fix, two-stage (L1 CPU + L2 GPU) pruning results. |
| [`docs/ao_cpu_gemm_benchmark.md`](docs/ao_cpu_gemm_benchmark.md) | ao (torchao) CPU GEMM 算子全面扫描：6 个可用算子对比、`_weight_int4pack_mm_for_cpu` 量化方法详解、int4 vs FP32 性能 benchmark、COMPASS 外推分析。 |
| [`docs/compass_perhead_scheduling_guide.md`](docs/compass_perhead_scheduling_guide.md) | COMPASS per-head KV cache 调度设计：独立 head 选择/传输/计算、staging buffer 内存布局、同步模型、m_global 跨 chunk bug 修复、IO 节省实测数据。 |

---

## 4. Configuration Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 1024 | Tokens per block (4096 supported). |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context. |
| `gpu_memory_utilization` | 0.9 | Target GPU memory fraction. |
| `enable_cpu_offload` | False | Enable for long context (required on 3090/4090). |
| `enforce_eager` | False | Set True to disable CUDA graphs (useful for debugging). |

---

**Author**: Zijie Tian / Gemini CLI
**Version**: 1.0 (Migrated from Claude Code)
