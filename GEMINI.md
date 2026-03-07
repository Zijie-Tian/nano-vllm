# GEMINI.md

This file provides foundational mandates and guidance for Gemini CLI when working with the Nano-vLLM repository.

## Project Overview

Nano-vLLM is a lightweight (~1,200 lines) implementation for fast offline LLM inference. It supports models like Qwen2/3, Llama-3, and GLM-4, featuring a specialized CPU offload system for long-context inference on consumer GPUs (e.g., RTX 3090/4090).

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
*   **PYTHONPATH**: Use `PYTHONPATH=$(pwd):$PYTHONPATH` instead of `pip install -e .` to ensure isolation between worktrees.
*   **test_ruler.py**: Read `docs/test_ruler_usage_guide.md` before running. Do not use `--help`. Match `data-dir` with appropriate `max-model-len`.
*   **Documentation Indexing**: Whenever a new document is added to the `docs/` directory, its path and purpose **MUST** be immediately indexed in both `GEMINI.md` and `CLAUDE.md`.
*   **Planning Files**: Use `findings.md` and `task_plan.md` for complex tasks. These are excluded from git. Clear old ones before starting a new task.

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
