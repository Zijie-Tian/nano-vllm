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
*   **Profiling (nsys)**: **MUST** use `scripts/profile_offload.sh`. Do NOT run `nsys` directly. **MUST** profile on a single GPU only — do NOT run other GPU workloads in parallel during profiling, as PCIe bus contention skews H2D/D2H transfer measurements.
*   **Benchmarking**: Before running `bench*.py`, ensure exclusive GPU access by checking for other compute-intensive PIDs.
*   **Single-GPU Testing (CRITICAL)**: When running `test_ruler.py` or `profile_offload.sh`, **NEVER** run two instances simultaneously on different GPUs. CPU-side operations (AVX cosine similarity in `cos matmul`, memory bandwidth for H2D staging) are shared resources — parallel GPU runs cause severe CPU contention (e.g., `cos matmul` ballooning from 0.08s to 1.8s, a 20x slowdown). Always run benchmarks **sequentially** on one GPU at a time.
*   **COMPASS Hyperparameters**: When testing COMPASS, always pass `--compass-top-p` and `--compass-lambda` via CLI. For correctness validation (100% density), use `--compass-top-p 1.0 --compass-lambda 1e-10`. For performance benchmarking, use `--compass-top-p 0.9 --compass-lambda 0.0001`.
*   **`lambda=0.0` Bug**: Never use `--compass-lambda 0.0` — it triggers a `ValueError: math domain error` due to `log(0)`. Use `1e-10` as the minimum value.

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
*   **test_ruler.py**: Read `docs/test_ruler_usage_guide.md` before running. Do not use `--help`. Match `data-dir` with appropriate `max-model-len`. **MANDATORY**: When tuning COMPASS hyperparameters (`top-p` and `lambda_threshold`), you MUST use the explicitly added CLI arguments (`--compass-top-p` and `--compass-lambda`) rather than modifying defaults in `config.py`.
    *   **Standard Testing Commands**: To quickly verify single-sample correctness, use these verified templates:
        *   **Full Context (no sparse)**: `source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py --model ~/models/Llama-3.1-8B-Instruct --data-dir tests/data/ruler_32k --datasets niah_single_1 --num-samples 1 --max-model-len 40960 --enable-offload`
        *   **BLASST**: `source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py --model ~/models/Llama-3.1-8B-Instruct --data-dir tests/data/ruler_32k --datasets niah_single_1 --num-samples 1 --max-model-len 40960 --enable-offload --sparse-policy BLASST`
        *   **COMPASS (correctness, 100% density)**: `source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py --model ~/models/Llama-3.1-8B-Instruct --data-dir tests/data/ruler_32k --datasets niah_single_1 --num-samples 1 --max-model-len 40960 --enable-offload --sparse-policy COMPASS --compass-top-p 1.0 --compass-lambda 1e-10`
        *   **COMPASS (performance, top-p=0.9)**: `source build/nano-vllm-envs.sh && CUDA_VISIBLE_DEVICES=<GPU> python3 tests/test_ruler.py --model ~/models/Llama-3.1-8B-Instruct --data-dir tests/data/ruler_32k --datasets niah_single_1 --num-samples 1 --max-model-len 40960 --enable-offload --sparse-policy COMPASS --compass-top-p 0.9 --compass-lambda 0.0001`
    *   **Data & Model Paths**: The model directory MUST be `~/models`, and RULER data MUST be in `tests/data`.
    *   **Interpreting COMPASS Logs**: The `select_blocks` output shows `IO_density` (union of blocks across heads needed for H2D transfer) and `Compute_density` (actual per-head sub-blocks computed vs full attention). The `compute_chunked_prefill` output shows how many intra-layer pipeline pieces are used for overlap.
    *   **Expected Performance (32k, Llama-3.1-8B, single 3090)**:
        *   Full context (no sparse): ~12s prefill
        *   COMPASS top-p=0.9: ~7-10s prefill (depends on sample sparsity)
        *   COMPASS top-p=1.0: ~10-11s prefill (all blocks selected, pipeline overhead only)
*   **Nsys Profiling with COMPASS**: Use `scripts/profile_offload.sh` with COMPASS-specific arguments:
    ```
    source build/nano-vllm-envs.sh && bash scripts/profile_offload.sh \
        --policy COMPASS --gpu <GPU> --ctx-len 32k \
        --dataset niah_single_1 \
        --model ~/models/Llama-3.1-8B-Instruct \
        --data-dir tests/data/ruler_32k/ \
        --compass-top-p 0.9 --compass-lambda 0.0001
    ```
    *   Output `.nsys-rep` files are saved to `results/nsys/`.
    *   In the Nsight Systems timeline, look for: orange="V2 Packed Gather" (CPU packing), green="V2 H2D Packed" (PCIe transfer), blue="V2 Jagged Kernel" (GPU compute). Overlap between green and blue across different slots confirms the intra-layer pipeline is working.

### 2.3 COMPASS Parameter Reference & Tuning Guide

**CLI Arguments** (passed to `test_ruler.py` or `profile_offload.sh`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--compass-top-p` | `0.9` (in config.py) | Top-p threshold for CPU L1 sub-block selection. Higher → more blocks selected → higher accuracy but slower. |
| `--compass-lambda` | `0.001` (in config.py) | Lambda threshold for block importance scoring. Controls minimum relevance cutoff via `log(lambda)`. |

**Internal Config** (in `nanovllm/config.py`, `SparsePolicyConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compass_top_p` | `0.9` | Same as `--compass-top-p` CLI. |
| `lambda_threshold` | `0.001` | Same as `--compass-lambda` CLI. |

**Tuning Recipes**:
*   **Correctness Validation** (verify pipeline logic without sparsity): `--compass-top-p 1.0 --compass-lambda 1e-10` → selects ALL blocks, 100% density, must achieve 100% accuracy. If this fails, it indicates a bug in the pipeline, not in the sparsity logic.
*   **Performance Benchmarking** (typical sparse regime): `--compass-top-p 0.9 --compass-lambda 0.0001` → ~89% compute density on 32k NIAH tasks, good accuracy/speed tradeoff.
*   **Aggressive Sparsity** (higher speedup, risk accuracy drop): `--compass-top-p 0.7 --compass-lambda 0.001` → significant pruning, may fail on multi-needle tasks.
*   **Sweep top-p**: Fix `--compass-lambda 0.0001`, sweep `--compass-top-p` from `0.5` to `1.0` in steps of `0.1` to find the accuracy/performance sweet spot.
*   ⚠️ **NEVER** use `--compass-lambda 0.0` → crashes with `ValueError: math domain error` due to `log(0)`. Use `1e-10` as minimum.

**Understanding Log Output**:
```
# select_blocks log (one per layer per seq_chunk):
[COMPASS] layer=5, seq_chunk=3: IO_density=92.0% (206/224), Compute_density=89.1% (1596/1792), per-head: [H0:200, H1:199, ...]

# compute_chunked_prefill log (one per layer per seq_chunk):
[COMPASS] layer=5, seq_chunk=3: total 1596 sub-blocks -> pipelined into 4 pieces (max 2 blks/piece) for overlap
```
*   **`IO_density`**: Percentage of KV blocks that need H2D transfer (union across all heads). Lower = less PCIe bandwidth used.
*   **`Compute_density`**: Percentage of sub-blocks actually computed by the GPU kernel (sum across all heads). Lower = faster Triton kernel.
*   **`pipelined into N pieces`**: Number of intra-layer pipeline chunks. More pieces = better overlap potential between H2D and GPU compute, but more kernel launch overhead.
*   **Documentation Indexing**: Whenever a new document is added to the `docs/` directory, its path and purpose **MUST** be immediately indexed in both `GEMINI.md` and `CLAUDE.md`.
*   **Planning Files**: Use `findings.md`, `task_plan.md`, and `progress.md` for complex tasks. These are excluded from git. **At the beginning of every new task, you MUST automatically delete any existing `task_plan.md`, `findings.md`, and `progress.md` files to ensure a fresh state.**

### 2.3 Monitoring
*   **GPU Monitoring**: For profiling or OOM debugging, run monitoring commands in the background. Prefer `nvidia-smi` queries or specialized profiling tools (nsys) directed to background output files.

### 2.4 Assistant Interaction Rules
*   **Image Generation**: 在我们的对话中，当提示词涉及到具体的物体、场景概念（例如“反重力”、“科幻设备”等），或者要求“展示”某个画面时，请务必直接调用图像生成工具生成实际的图像。在这些情况下，绝对不要使用 Mermaid.js 或代码块来绘制图表，除非明确在提示词中使用了“流程图”、“架构图”或“Mermaid”等词汇。
*   **Default Rule Scope**: 除非我明确指定，否则以后要求添加的新规则，请默认添加到当前项目的 `GEMINI.md` 中，而不是全局 `~/.gemini/GEMINI.md`。

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
| [`docs/compass_async_pipeline_guide.md`](docs/compass_async_pipeline_guide.md) | COMPASS async double-buffered pipeline：双 slot 交替、coalesced gather、同步模型、NVTX 标记、性能对比（prefill −47%）。 |
| [`docs/compass_head_first_layout_and_async_staging.md`](docs/compass_head_first_layout_and_async_staging.md) | COMPASS Head-First KV Cache layout eliminating H2D IO amplification, and fully asynchronous multi-staging buffers avoiding CPU blocking overhead. |
| [`docs/compass_v2_jagged_architecture.md`](docs/compass_v2_jagged_architecture.md) | COMPASS V2 Jagged Memory Architecture: 1D packed buffer mapping, top-p slice gathering, bulk DMA, and custom Triton kernel integration. |
| [`docs/compass_l2_pruning_and_kernel_optimization.md`](docs/compass_l2_pruning_and_kernel_optimization.md) | COMPASS L2 dynamic pruning (BLASST-style GPU skip) and pipeline kernel overhead reduction: in-place merge kernel, mask_buffer pre-allocation, density tracking. |

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
