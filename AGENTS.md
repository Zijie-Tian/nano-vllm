# Feynman Project Guide: Nano-vLLM

This file is read automatically at startup. It is the durable project memory for Feynman.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3, Llama-3, and GLM-4 models with CPU offload for long-context inference.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture_guide.md`](docs/architecture_guide.md) | Core components, CPU offload system design, ring buffer architecture, stream configuration |
| [`docs/sparse_policy_architecture.md`](docs/sparse_policy_architecture.md) | SparsePolicy abstraction: prefill/decode delegation, pipeline modes, policy implementations |
| [`docs/rope_policy_design.md`](docs/rope_policy_design.md) | PREROPE / POSTROPE 语义设计：RoPE 放置位置、KV cache/offload 语义、以及与 FULL 对齐所需的同步约束。 |
| [`docs/postrope_sparge_chunked_prefill_design.md`](docs/postrope_sparge_chunked_prefill_design.md) | 当前 Sparge-style POSTROPE 设计：独立 policy、Q/K fix-block 适配、dense decode、RULER 主线命令与验证结果。 |
| [`docs/sparse_policy_implementation_guide.md`](docs/sparse_policy_implementation_guide.md) | How to implement custom SparsePolicy: required methods, hooks, ring buffer pipeline pattern |
| [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md) | Block sparse attention methods (XAttention, FlexPrefill, MInference, AvgPool, Quest), computation flow, algorithms |
| [`docs/xattention_algorithm_guide.md`](docs/xattention_algorithm_guide.md) | XAttention 算法详解: stride reshape、Triton kernels、BSA 依赖、块选择算法 |
| [`docs/xattn_kernels_guide.md`](docs/xattn_kernels_guide.md) | XAttention Triton kernels: flat_group_gemm (反对角线求和)、softmax_fuse_block_sum (block 聚合) |
| [`docs/xattn_kv_chunking_kernels.md`](docs/xattn_kv_chunking_kernels.md) | XAttention KV Chunking: 三阶段 softmax、存储开销分析 (O(S) vs O(S²))、峰值显存优化 (8x)、Q/KV 独立分块 |
| [`docs/xattn_chunked_prefill.md`](docs/xattn_chunked_prefill.md) | XAttention chunked prefill: API、使用方式、一致性要求 |
| [`docs/xattn_bsa_policy_design.md`](docs/xattn_bsa_policy_design.md) | XAttention BSA Policy: 算法设计、性能基准(128K)、内存管理、density 统计 |
| [`docs/xattn_density_benchmark.md`](docs/xattn_density_benchmark.md) | 📊 XAttention Density Benchmark: 4K-32K context、stride 参数、per-layer density 分析 |
| [`docs/block_sparse_attn_interface.md`](docs/block_sparse_attn_interface.md) | BSA (Block Sparse Attention) 接口文档: 函数签名、使用示例、约束条件 |
| [`docs/debugging_guide.md`](docs/debugging_guide.md) | PyTorch hooks for debugging, hook positions, tensor comparison, memory profiling |
| [`docs/optimization_guide.md`](docs/optimization_guide.md) | Performance optimizations: sgDMA (15x), Triton merge (4.3x), N-way pipeline (2x) |
| [`docs/known_issues.md`](docs/known_issues.md) | Documented bugs and fixes: partial last block bug, block size 4096 race condition |
| [`docs/ruler_benchmark_results_32k.md`](docs/ruler_benchmark_results_32k.md) | RULER benchmark results (32K context): 13 tasks, 92.3% accuracy, CPU offload performance |
| [`docs/ruler_32k_chunked_offload_issue.md`](docs/ruler_32k_chunked_offload_issue.md) | ⚠️ OPEN ISSUE: 32K chunked offload accuracy problem (20% error rate in RULER) |
| [`docs/chunked_attention_solutions.md`](docs/chunked_attention_solutions.md) | 🔧 SOLUTIONS: Chunked attention 准确性问题的代码分析和解决方案 |
| [`docs/nsys_wrong_event_order_bug.md`](docs/nsys_wrong_event_order_bug.md) | 🐛 NSYS BUG: Ring buffer pipeline 触发 nsys 时间戳乱序问题的调试记录 |
| [`docs/cpu_scheduling_latency_analysis.md`](docs/cpu_scheduling_latency_analysis.md) | ⚡ PERF: CPU 调度延迟分析，kernel 间隙来源，GPU 利用率优化方向 |
| [`docs/bench_offload_results.md`](docs/bench_offload_results.md) | 📊 BENCH: CPU offload 性能测试结果，Full vs XAttention 对比 (32K/128K) |
| [`docs/cpu_offload_optimization_strategies.md`](docs/cpu_offload_optimization_strategies.md) | 🚀 OPT: CPU offload 优化策略：chunk size、CUDA Graph、前沿研究(InfiniGen/ShadowKV) |
| [`docs/gpu_only_xattn_guide.md`](docs/gpu_only_xattn_guide.md) | 🚀 GPU-Only XAttention: 内存预分配、性能分析 (32K +15%, 64K +41%)、CUDA Graph 限制 |
| [`docs/xattn_performance_analysis.md`](docs/xattn_performance_analysis.md) | 📊 XAttention 性能分析: NVTX 标记、block size 影响、estimate vs compute 耗时对比 |
| [`docs/observer_architecture.md`](docs/observer_architecture.md) | 📊 Observer 架构: InferenceObserver (TTFT/TPOT)、MemoryObserver (H2D/D2H/D2D) 设计 |
| [`docs/memory_communication_benchmark.md`](docs/memory_communication_benchmark.md) | 📊 通信量测试: Full vs XAttention 通信量对比 (32K/64K)、阶段分离统计 |
| [`docs/estimate_block_size_performance.md`](docs/estimate_block_size_performance.md) | 🔥 PERF: estimate 阶段 block_size 性能分析，softmax_fuse_block_sum 最优点 (512-1024)，当前 4096 慢 15x |
| [`docs/long_context_models_1m.md`](docs/long_context_models_1m.md) | 📚 REF: 1M+ 上下文长度模型列表 (Qwen/GLM/InternLM/Llama/VL)，≤10B 推荐模型 |
| [`docs/new_model_integration_guide.md`](docs/new_model_integration_guide.md) | 🔧 GUIDE: 新模型整合指南 - 配置映射、RoPE变体、EOS处理、权重转换、验证清单 |
| [`docs/xattn_density_alignment_analysis.md`](docs/xattn_density_alignment_analysis.md) | 📊 ANALYSIS: GPU-only vs Offload 模式 density 对齐分析，chunked softmax 边界效应，5-7% 差异根因 |
| [`docs/xattn_kv_chunking_density_test.md`](docs/xattn_kv_chunking_density_test.md) | 🧪 TEST: XAttention KV chunking density 验证，threshold=1.0 对齐，threshold<1.0 差异 10-13% |
| [`docs/gpuonly_density_alignment_test.md`](docs/gpuonly_density_alignment_test.md) | ✅ TEST: Density 对齐验证 (GPU-only + Offload, 4K-64K)，xattn_estimate vs KV chunking 完全一致 |
| [`docs/xattn_memory_benchmark.md`](docs/xattn_memory_benchmark.md) | 📊 BENCH: XAttention 内存基准测试，Qwen3-0.6B 32K 在 24GB 显存可行 (gpu-util=0.28) |
| [`docs/xattn_offload_stream_sync_fix.md`](docs/xattn_offload_stream_sync_fix.md) | 🐛 FIX: XAttention Offload stream 同步 bug，Pass1/Pass2 K 数据不一致，compute_stream 包装 |
| [`docs/xattn_density_types.md`](docs/xattn_density_types.md) | 📊 Compute vs Comm density: BSA block (128) vs CPU block (4096) 粒度，聚合效应导致 comm=100% |
| [`docs/xattn_density_alignment_verification.md`](docs/xattn_density_alignment_verification.md) | ✅ VERIFIED: GPU-only vs Offload density 对齐验证 (32K 差异 0.37%, 64K 差异 0.09%) |
| [`docs/xattn_768k_density_benchmark.md`](docs/xattn_768k_density_benchmark.md) | 📊 BENCH: 768K context density 测试，compute=56.95%, comm=100%，聚合效应分析 |
| [`docs/test_ruler_usage_guide.md`](docs/test_ruler_usage_guide.md) | 📖 GUIDE: test_ruler.py 使用指南，RULER benchmark 测试命令，已验证的命令示例 |
| [`docs/ruler_rope_policy_alignment.md`](docs/ruler_rope_policy_alignment.md) | ✅ TEST: FULL / POSTROPE / PREROPE 在 offload + chunked prefill 下的 RULER 五样本逐文本/逐 token 对齐记录（GPU0/GPU1）。 |
| [`docs/xattn_offload_profiling_32k.md`](docs/xattn_offload_profiling_32k.md) | 📊 PROFILE: XAttn vs Full 32K nsys 分析，estimate 占 41%，find_blocks 占 37%，compute 仅 21% |
| [`docs/select_blocks_ring_buffer_pipeline.md`](docs/select_blocks_ring_buffer_pipeline.md) | ⚡ PERF: select_blocks ring buffer pipeline 优化，128K prefill -9.5%，find_blocks -44.8% |
| [`docs/changelog_2026-02-05.md`](docs/changelog_2026-02-05.md) | 📋 CHANGELOG: GQA buffer OOM 修复 (节省 16GB)，tests 目录清理 (-4306 行) |
| [`docs/rope_data_collection_guide.md`](docs/rope_data_collection_guide.md) | 📊 DATA: RoPE 数据收集流程，pre/post RoPE QKV 保存、chunk 合并、上传备份，三模型七种 context |
| [`docs/sparse_attention_blasst.md`](docs/sparse_attention_blasst.md) | ⚡️ BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding |
| [`docs/blasst_performance_analysis.md`](docs/blasst_performance_analysis.md) | 📊 BLASST Performance: Density tracking and parameter sensitivity |
| [`docs/blasst_mask_visualization_guide.md`](docs/blasst_mask_visualization_guide.md) | 🗺️ BLASST Visualization: Guide to export and plot attention masks |
| [`docs/trtllm_skip_softmax_implementation_details.md`](docs/trtllm_skip_softmax_implementation_details.md) | 🔍 DEEP DIVE: Implementation details of Skip Softmax (BLASST) in TensorRT-LLM |
| [`docs/sparse_policy_offload_control_guide.md`](docs/sparse_policy_offload_control_guide.md) | 🔧 GUIDE: How to control KV cache CPU offloading from custom SparsePolicies |
| [`docs/compass_tmac_verification_design.md`](docs/compass_tmac_verification_design.md) | 📊 VERIFICATION: Design for COMPASSPolicy TMAC accuracy verification data pipeline |
| [`docs/tmac_tvm_qgemm_migration_guide.md`](docs/tmac_tvm_qgemm_migration_guide.md) | 🔧 GUIDE: T-MAC QGeMM TVM integration, dependencies stripping, and alignment tests |
| [`docs/blasst_bugfix_mglobal_causal.md`](docs/blasst_bugfix_mglobal_causal.md) | 🐛 BUGFIX: BLASST m_global/LSE decoupling & element-wise causal masking |
| [`docs/blasst_perhead_density_analysis.md`](docs/blasst_perhead_density_analysis.md) | 📊 ANALYSIS: Per-head KV density analysis: GQA IO implications for sparse attention offload |
| [`docs/blasst_density_collection_guide.md`](docs/blasst_density_collection_guide.md) | 📊 GUIDE: BLASST density 数据采集流程: `test_ruler.py` 采集 → `scripts/export_blasst_density_csv.py` 导出 CSV → `scripts/analyze_blasst_density.py` 汇总分析 |
| [`docs/compass_tmac_l1_batching.md`](docs/compass_tmac_l1_batching.md) | 深入解析 COMPASS TMAC 中的 L1 批量化优化：数学约束、工程实现以及针对多头注意力的 Python 端瓶颈分析。 |
| [`docs/compass_torch_gemm_migration.md`](docs/compass_torch_gemm_migration.md) | COMPASS TMAC to Torch CPU GEMM migration: BMM batching optimization, API simplification, and IO reduction benchmark results (32K context). |
| [`docs/ao_cpu_gemm_benchmark.md`](docs/ao_cpu_gemm_benchmark.md) | ao (torchao) CPU GEMM 算子全面扫描：6 个可用算子对比、`_weight_int4pack_mm_for_cpu` 量化方法详解、int4 vs FP32 性能 benchmark、COMPASS 外推分析。 |
| [`docs/compass_head_first_layout_and_async_staging.md`](docs/compass_head_first_layout_and_async_staging.md) | COMPASS Head-First KV Cache layout eliminating H2D IO amplification, and fully asynchronous multi-staging buffers avoiding CPU blocking overhead. |
| [`docs/compass_v2_jagged_architecture.md`](docs/compass_v2_jagged_architecture.md) | COMPASS V2 Jagged Memory Architecture: 1D packed buffer mapping, top-p slice gathering, bulk DMA, and custom Triton kernel integration. |

## Rules Index

| Rule | Purpose |
|------|---------|
| [`.feynman/rules/multi-gpu-debugging.md`](.feynman/rules/multi-gpu-debugging.md) | **Multi-GPU debugging**: GPU allocation (1-2 for validation, rest for exploration), single-task validation policy |
| [`.feynman/rules/gpu-testing.md`](.feynman/rules/gpu-testing.md) | GPU type detection, card assignment, needle test requirements |
| [`.feynman/rules/sparse-policy.md`](.feynman/rules/sparse-policy.md) | SparsePolicy implementation requirements |
| **`CHANGELOG.md`** (Feynman Native) | Feynman uses `CHANGELOG.md` for task ledgers instead of Git-ignored `.feynman/rules/planning-with-files.md`. |
| **Feynman `process` tool** | Replaces `.feynman/rules/gpu-monitor.md`. Run monitors via `process` instead of blocking shell loops. |
| [`.feynman/rules/test-ruler.md`](.feynman/rules/test-ruler.md) | **test_ruler.py 规则**: 禁止 --help，必须查阅文档，含快速参考和命令模板 |
| [`.feynman/rules/doc-sync.md`](.feynman/rules/doc-sync.md) | **Documentation Sync Rule**: 强制规定 `docs/` 下的任何变更必须同步更新所有 `*.md` 索引 (AGENTS/CLAUDE/GEMINI) |

## Multi-Instance Development & GPU Mutex

### 1. Benchmarks (`bench*.py`) - Exclusive GPU Access Required
Before running any `bench*.py` script, you MUST wait for exclusive GPU access:
```bash
# Check and wait for GPU to be free
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "GPU busy, waiting 10s..."
  sleep 10
done
```

### 2. Other Scripts (tests, examples) - No Special Requirements
For non-benchmark scripts, exclusive GPU access is NOT required. Multiple processes can run simultaneously on different GPUs.

### 3. PYTHONPATH Isolation
When running multiple agent instances on different worktrees, do NOT use `pip install -e .` globally. Use `PYTHONPATH` directly:
```bash
PYTHONPATH=/home/tzj/Code/nano-vllm:$PYTHONPATH python <script.py>
```

## Configuration & Constraints

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 1024 | Tokens per block (4096 now works after race condition fix) |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |
| `enforce_eager` | False | Set True to disable CUDA graphs |

### Benchmarking Rules

**GPU-only 测试模型选择**:
| GPU | 显存 | GPU-only 测试模型 |
|-----|------|------------------|
| RTX 3090 | 24GB | **Qwen3-0.6B** (必须，7B+ 模型会 OOM) |
| A100 | 40GB+ | Qwen3-0.6B / 4B / 7B 均可 |

**Offload Mode Constraint**: When using `enable_cpu_offload=True`, only test with context length ≥ 32K. Shorter contexts don't exercise the chunked offload pipeline properly.

**Common Issues**:
1. `max_num_batched_tokens < max_model_len`: Set equal for long context
2. CUDA graph dimension mismatch: Ensure `input_len + output_len <= max_model_len`
3. RoPE out of bounds: Check model's `max_position_embeddings` in config.json

**Model Limits**:
- Qwen3-0.6B/4B: 40960 tokens
- Qwen2.5-7B-Instruct-1M: 1048576 tokens

**Performance (Qwen3-0.6B)**:
- GPU: ~18k tok/s (prefill), ~100 tok/s (decode)
- CPU Offload (16K): ~14k tok/s (prefill)
- CPU Offload (32K): ~13k tok/s (prefill)


## Appendix: Full Rules and Context from .feynman/rules/


### File: .feynman/rules/agent-result-format.md
# Agent Result Format Rules

## Purpose

Minimize token usage when background agents return results to the main agent. Raw program output is verbose and wastes context window space.

---

## 1. Result Formatting Principle

**MUST** return **structured summaries** instead of raw output.

| Don't | Do |
|-------|-----|
| Full program stdout/stderr | Key metrics only |
| Debug logs | Pass/Fail status |
| Verbose error stacks | Error summary + location |

---

## 2. Standard Result Templates

### 2.1 Test Results (RULER, Unit Tests, etc.)

```markdown
## Test Results: [Task Name]

**Pass Rate**: X / Y (Z%)

### Failed Samples (if any)
| Sample | Expected | Got |
|--------|----------|-----|
| N | expected_value | actual_value |

### Passed Samples
[List sample IDs or "All N samples passed"]
```

**Example** (instead of raw test output):
```markdown
## Test Results: niah_single_1 (Samples 0-49)

**Pass Rate**: 50 / 50 (100%)

### Passed Samples
All 50 samples passed.
```

### 2.2 Benchmark Results

```markdown
## Benchmark Results: [Task Name]

| Metric | Value |
|--------|-------|
| Throughput | X tok/s |
| Latency (p50) | Y ms |
| Latency (p99) | Z ms |
| Memory Peak | W GB |
```

### 2.3 Build/Compile Results

```markdown
## Build Results: [Target]

**Status**: SUCCESS / FAILED

### Errors (if any)
| File | Line | Error |
|------|------|-------|
| path/to/file.py | 123 | error message |
```

### 2.4 Investigation/Research Results

```markdown
## Investigation: [Topic]

### Findings
1. Finding 1 (with file:line reference)
2. Finding 2

### Relevant Files
- path/to/file1.py: description
- path/to/file2.py: description

### Conclusion
[1-2 sentence summary]
```

---

## 3. Mandatory Fields by Task Type

| Task Type | Required Fields |
|-----------|-----------------|
| Test Run | Pass/Fail count, failed sample details |
| Benchmark | Key metrics (throughput, latency, memory) |
| Build | Status, error locations |
| Search | File paths, line numbers, brief context |
| Verification | Before/After comparison, conclusion |

---

## 4. What to EXCLUDE

**MUST NOT** include in results:

| Exclude | Reason |
|---------|--------|
| Full stack traces | Extract error type + location only |
| Model loading logs | Not relevant to result |
| Progress bars / tqdm output | Noise |
| Warnings (unless critical) | Noise |
| Repeated successful outputs | "All X passed" is sufficient |
| Timestamps | Usually not needed |
| Device info (unless debugging hardware) | Noise |

---

## 5. Agent Prompt Template

When spawning background agents, include this instruction:

```
When reporting results, use a structured summary format:
- For tests: Pass rate, failed sample details (expected vs actual)
- For benchmarks: Key metrics table
- Do NOT include raw program output, logs, or verbose debug info
- Focus on actionable information only
```

---

## 6. Main Agent Instructions

When spawning a background agent for testing:

**Before** (verbose):
```
Run tests for samples 0-49 and report the output.
```

**After** (structured):
```
Run tests for samples 0-49. Report results as:
- Total pass/fail count
- For each failure: sample ID, expected value, actual value
- Do NOT include raw program output or logs
```

---

## 7. Examples

### Bad (Wastes ~500 tokens):
```
The test output was:
Loading model from ~/models/Llama-3.1-8B-Instruct...
Model loaded in 12.3s
[niah_single_1] Sample 0: PASS | Expected: 1234567 | Got: : 1234567.<|eot_id|>
[niah_single_1] Sample 1: PASS | Expected: 2345678 | Got: : 2345678.<|eot_id|>
... (50 more lines) ...
```

### Good (Uses ~50 tokens):
```
## Test Results: niah_single_1 (Samples 0-49)

**Pass Rate**: 50 / 50 (100%)

All samples passed.
```

---

## 8. Token Savings Estimate

| Result Type | Raw Output | Structured | Savings |
|-------------|------------|------------|---------|
| 50-sample test | ~1000 tokens | ~100 tokens | 90% |
| Benchmark run | ~500 tokens | ~80 tokens | 84% |
| Build failure | ~2000 tokens | ~200 tokens | 90% |

---

## 9. Integration

This rule should be applied when:
1. Spawning agents via Task tool
2. Running background commands
3. Processing results from completed agents

Combine with `multi-gpu-debugging.md` for efficient parallel testing workflows.



### File: .feynman/rules/code-analysis.md
# Code Analysis

## Use cclsp MCP for Code Navigation

When analyzing code, understanding call chains, or exploring the codebase, **prefer using the cclsp MCP tools** over grep/glob-based searches:

### Available cclsp Tools

| Tool | Purpose |
|------|---------|
| `mcp__cclsp__find_definition` | Jump to symbol definition |
| `mcp__cclsp__find_references` | Find all usages of a symbol |
| `mcp__cclsp__rename_symbol` | Rename a symbol across the codebase |
| `mcp__cclsp__get_diagnostics` | Get LSP diagnostics (errors, warnings) |
| `mcp__cclsp__restart_server` | Restart the LSP server if needed |

### When to Use cclsp

1. **Understanding call chains**: Use `find_references` to trace how functions are called
2. **Finding implementations**: Use `find_definition` to jump to actual code
3. **Refactoring**: Use `rename_symbol` for safe cross-file renames
4. **Code quality**: Use `get_diagnostics` to check for issues

### Example Workflow

```
1. User asks: "How does the prefill flow work?"
2. Use find_definition to locate key entry points (e.g., run_chunked_offload_prefill)
3. Use find_references to trace the call chain through the codebase
4. Read relevant code sections to understand the implementation
```

### Benefits over grep/glob

- **Semantic understanding**: cclsp understands code structure, not just text patterns
- **Accurate references**: Finds actual usages, not just text matches
- **Cross-file navigation**: Follows imports and definitions across modules
- **Type-aware**: Understands Python types and class hierarchies



### File: .feynman/rules/commands.md
# Commands

## Running (with PYTHONPATH)

For multi-instance development, use PYTHONPATH instead of pip install:

```bash
# Run example
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python example.py

# Run benchmarks
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python bench.py
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python bench_offload.py
```

## Config Defaults

- `max_num_batched_tokens`: 16384
- `max_num_seqs`: 512
- `kvcache_block_size`: 4096
- `gpu_memory_utilization`: 0.9
- `enforce_eager`: False (enables CUDA graphs)



### File: .feynman/rules/complex-code-generation.md
# Complex Code Generation Rule

## Purpose

This rule governs the use of Codex MCP for complex code generation tasks including Triton kernel design, CUDA programming, and SIMD optimization. The goal is to leverage Codex's specialized capabilities for low-level performance code while maintaining quality through iterative testing and feedback.

---

## When to Apply This Rule

**MUST** use Codex MCP when implementing:

| Task Type | Examples |
|-----------|----------|
| Triton Kernels | Custom attention kernels, fused operations, block-sparse kernels |
| CUDA Programming | Custom CUDA kernels, kernel fusion, memory optimization |
| SIMD Optimization | AVX/AVX2/AVX-512 vectorized code, NEON intrinsics |
| Low-level Optimizations | Block-wise algorithms, warp-level primitives, shared memory tuning |
| Performance-critical Code | Kernel auto-tuning, pipeline optimization, memory coalescing |

---

## Mandatory Requirements

### 1. Model Specification (CRITICAL)

**MUST** always use `gpt-5.4` model:

```python
# Correct: Explicit model specification
mcp__codex-cli__codex(
    prompt="...",
    model="gpt-5.4",
    ...
)
```

### 2. Codex MCP Availability Check

Before starting complex code tasks, verify Codex MCP is available:

```python
# Check connectivity
mcp__codex-cli__ping(message="ping")
```

If Codex MCP is unavailable:
1. Fall back to manual implementation
2. Document the fallback in findings
3. Consider retrying Codex later for comparison

---

## Workflow

### Phase 1: Requirements Specification

**My Responsibility**: Provide comprehensive design specifications to Codex

Required specification components:

```markdown
## Design Specification Template

### 1. Algorithm Overview
- What: High-level algorithm description
- Why: Performance motivation and expected gains
- Where: Integration point in existing codebase

### 2. Interface Definition
```python
# Expected function signature
def kernel_name(input1: torch.Tensor, input2: torch.Tensor, ...) -> torch.Tensor:
    """
    Args:
        input1: [shape] dtype, description
        input2: [shape] dtype, description
    Returns:
        output: [shape] dtype, description
    """
```

### 3. Constraints & Requirements
| Category | Requirement |
|----------|-------------|
| Block size | Must be power of 2, preferably 128-512 |
| Memory | Max shared memory per block: XX KB |
| Precision | FP16/BF16/FP32 requirements |
| Numerical | Tolerance for numerical errors |

### 4. Algorithm Steps
1. Step 1: Load data from global memory
2. Step 2: Compute intermediate results
3. Step 3: Store results back

### 5. Reference Implementation (if available)
```python
# Naive PyTorch reference for validation
def reference_impl(...):
    ...
```

### 6. Performance Targets
- Minimum speedup vs baseline: X%
- Memory bandwidth utilization: Y%
- Maximum latency: Z us
```

### Phase 2: Code Generation via Codex

**Codex's Responsibility**: Generate optimized code based on specifications

```python
mcp__codex-cli__codex(
    prompt="""
    [Insert complete Design Specification from Phase 1]

    Please implement a Triton kernel that:
    1. Follows the algorithm steps exactly
    2. Optimizes for the specified block size
    3. Handles edge cases (partial blocks, boundary conditions)
    4. Includes detailed comments explaining the optimization strategy
    5. Provides both the kernel and a Python wrapper function

    Output format:
    - Complete, runnable Python code
    - Type hints where appropriate
    - Documentation strings
    - Example usage
    """,
    model="gpt-5.4",
    reasoningEffort="high",  # Use high effort for complex kernels
    sandbox="workspace-write"
)
```

### Phase 3: Testing & Validation

**My Responsibility**: Test the generated code thoroughly

Mandatory test categories:

| Test Type | Purpose | Criteria |
|-----------|---------|----------|
| Correctness | Verify output matches reference | Max error < tolerance |
| Shape Coverage | Test various input dimensions | Edge cases (1, small, large, non-power-of-2) |
| Numerical Stability | Check for NaN/Inf | No invalid values |
| Performance | Measure speedup | Meet or exceed targets |
| Memory Safety | Check for OOB access | No CUDA errors |

Test script template:

```python
"""
Test: [Kernel Name]

Validation suite for Codex-generated Triton kernel.
"""
import torch
import triton
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

from nanovllm.ops.[kernel] import [kernel_func]

# ============================================================
# Parameters
# ============================================================
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_SHAPES = [
    (128, 64),      # Small
    (1024, 1024),   # Medium
    (8192, 8192),   # Large
    (1000, 500),    # Non-power-of-2
]
ATOL = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}

# ============================================================
# Reference Implementation
# ============================================================
def reference_fn(x, y):
    """PyTorch reference for validation."""
    return ...

# ============================================================
# Correctness Tests
# ============================================================
def test_correctness():
    for dtype in TEST_DTYPES:
        for shape in TEST_SHAPES:
            x = torch.randn(shape, dtype=dtype, device="cuda")
            y = torch.randn(shape, dtype=dtype, device="cuda")

            expected = reference_fn(x, y)
            actual = kernel_func(x, y)

            atol = ATOL[dtype]
            assert torch.allclose(actual, expected, atol=atol), \
                f"Failed: dtype={dtype}, shape={shape}"

    print(f"test_correctness: PASSED")

# ============================================================
# Performance Benchmark
# ============================================================
def benchmark():
    # Warmup
    for _ in range(10):
        ...

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        ...
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / 100
    print(f"Benchmark: {elapsed_ms:.3f} ms/iter")

if __name__ == "__main__":
    test_correctness()
    benchmark()
```

### Phase 4: Feedback Loop

**My Responsibility**: Provide detailed feedback to Codex for refinement

If tests fail, use this feedback template:

```markdown
## Test Feedback for Refinement

### Issue Summary
- Test: [Which test failed]
- Input: [Shape, dtype, configuration]
- Error: [Error message or numerical deviation]

### Expected Behavior
[What the correct output should be]

### Actual Behavior
[What the current code produces]

### Root Cause Analysis (if known)
[Your analysis of why it fails]

### Suggested Fix (if known)
[Specific suggestions for Codex]

### Current Code
```python
[paste relevant code section]
```
```

Then request refinement:

```python
mcp__codex-cli__codex(
    prompt="""
    The previous kernel has issues. Please fix:

    [Insert Feedback from above]

    Please provide the corrected implementation.
    """,
    model="gpt-5.4",
    reasoningEffort="high",
    sandbox="workspace-write"
)
```

---

## Code Quality Standards

### Generated Code Must Include

| Element | Requirement |
|---------|-------------|
| Docstrings | Full description of inputs, outputs, behavior |
| Comments | Explain non-obvious optimizations |
| Type Hints | For all public functions |
| Error Handling | Validate inputs, raise meaningful errors |
| Edge Cases | Handle empty tensors, boundary conditions |
| Numerical Stability | Use appropriate accumulation types |

### Triton-specific Requirements

```python
# Good Triton kernel characteristics:

@triton.jit
def good_kernel(
    input_ptr, output_ptr,
    stride_m, stride_n,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Clear description of what this kernel does.

    Args:
        input_ptr: Pointer to input tensor [M, N]
        output_ptr: Pointer to output tensor [M, N]
        ...
    """
    # 1. Program ID calculation with clear variable names
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Block-level offset calculation
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 3. Mask creation for boundary handling
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 4. Memory coalesced loads
    input_block = tl.load(input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask)

    # 5. Computation
    output_block = ...

    # 6. Guarded stores
    tl.store(output_ptr + ..., output_block, mask=mask)
```

---

## Integration with Existing Codebase

### File Organization

Place generated kernels in appropriate directories:

```
nanovllm/ops/
├── triton_kernels/     # Pure Triton kernels
│   ├── __init__.py
│   ├── attention.py
│   ├── softmax.py
│   └── [new_kernel].py
├── cuda/               # Custom CUDA kernels
│   ├── __init__.py
│   └── [kernel].cu
└── simd/               # SIMD optimized code
    ├── __init__.py
    └── [impl].py
```

### Registration Pattern

```python
# In nanovllm/ops/__init__.py

def get_kernel_impl(name: str):
    """Get optimal kernel implementation for current hardware."""
    if torch.cuda.is_available() and name in TRITON_KERNELS:
        return TRITON_KERNELS[name]
    return FALLBACK_KERNELS[name]
```

---

## Example: Complete Workflow

### Task: Implement Block-Sparse Attention Kernel

**Step 1**: Prepare specification
```markdown
## Design Spec: Block-Sparse Attention Kernel

### Algorithm
Compute attention only for selected block pairs...

### Interface
```python
def block_sparse_attention(
    q: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    k: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    v: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    block_indices: torch.Tensor,  # [batch, num_heads, num_blocks, 2]
    block_size: int = 128,
) -> torch.Tensor:            # [batch, num_heads, seq_len, head_dim]
```

### Constraints
- BLOCK_SIZE must be multiple of 64
- head_dim must be 64 or 128
- seq_len must be divisible by BLOCK_SIZE
```

**Step 2**: Call Codex MCP
```python
mcp__codex-cli__codex(
    prompt="[specification from Step 1]",
    model="gpt-5.4",
    reasoningEffort="high",
    sandbox="workspace-write"
)
```

**Step 3**: Create test file
```python
# tests/test_block_sparse_attention.py
# [Test implementation following template]
```

**Step 4**: Run tests
```bash
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_block_sparse_attention.py
```

**Step 5**: If failures, provide feedback and iterate

---

## Performance Validation Checklist

Before accepting generated code:

- [ ] Correctness: Matches reference within tolerance
- [ ] Numerical: No NaN/Inf for any valid input
- [ ] Memory: No OOB accesses detected by compute-sanitizer
- [ ] Performance: Meets speedup targets
- [ ] Integration: Works with existing pipeline
- [ ] Documentation: Fully documented API
- [ ] Edge cases: Handles all boundary conditions

---

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Vague requirements | Provide specific shapes, dtypes, constraints |
| Skip reference implementation | Always provide PyTorch reference for validation |
| Accept without testing | Run full test suite before integration |
| Ignore numerical errors | Investigate and fix precision issues |
| Skip edge case handling | Test and handle boundary conditions |
| Use default model | Explicitly specify `gpt-5.4` |

---

## Summary

| Phase | Responsible | Action |
|-------|-------------|--------|
| 1. Spec | Me | Write detailed design specification |
| 2. Generate | Codex (gpt-5.4) | Generate optimized kernel code |
| 3. Test | Me | Run correctness and performance tests |
| 4. Feedback | Me → Codex | Iterate until all tests pass |
| 5. Integrate | Me | Merge into codebase with proper structure |

---

**Author**: Zijie Tian
**Version**: 1.0



### File: .feynman/rules/doc-management.md
# Documentation Management

## CLAUDE.md Content Policy

**CLAUDE.md should only contain operational requirements:**
- Environment setup (PYTHONPATH, GPU mutex)
- Execution requirements (how to run tests/benchmarks)
- Quick configuration reference
- Documentation index (links to detailed docs)

**Technical details should go to docs/:**
- Architecture and design explanations
- Implementation details and code flows
- Debugging techniques
- Memory analysis and profiling
- Algorithm explanations

## When Adding New Technical Content

Follow this workflow:

### Step 1: Analyze and Document

If doing technical analysis (e.g., memory profiling):
1. Calculate theoretical values using formulas
2. Run actual tests to measure real values
3. Compare theoretical vs actual (expect < 10% error for valid models)
4. Document findings with both theory and empirical validation

### Step 2: Create/Update docs/

Create a new doc or update existing one in `docs/`:
```
docs/
├── architecture_guide.md      # Core components, design, flows
├── sparse_attention_guide.md  # Sparse attention methods
├── layerwise_offload_memory_analysis.md  # Memory analysis
├── debugging_guide.md         # Debugging techniques
└── <new_topic>_guide.md       # New technical topic
```

### Step 3: Update CLAUDE.md Documentation Index

Add entry to the Documentation Index table:
```markdown
| Document | Purpose |
|----------|---------|
| [`docs/new_doc.md`](docs/new_doc.md) | Brief description |
```

### Step 4: Refactor if Needed

If CLAUDE.md grows too large (> 150 lines), refactor:
1. Identify technical details that can be moved
2. Create appropriate doc in docs/
3. Replace detailed content with reference link
4. Keep only operational essentials in CLAUDE.md

## Documentation Structure Template

For new technical docs:

```markdown
# Topic Guide

Brief overview of what this document covers.

## Section 1: Concepts
- Key concepts and terminology

## Section 2: Implementation
- Code locations
- Key methods/functions

## Section 3: Details
- Detailed explanations
- Code examples

## Section 4: Validation (if applicable)
- Theoretical analysis
- Empirical measurements
- Comparison table
```

## Memory Analysis Template

When documenting memory behavior:

```markdown
## Theoretical Calculation

| Component | Formula | Size |
|-----------|---------|------|
| Buffer X | `param1 × param2 × dtype_size` | X MB |

## Empirical Validation

| Metric | Theoretical | Actual | Error |
|--------|-------------|--------|-------|
| Peak memory | X GB | Y GB | Z% |

## Key Findings
1. Finding 1
2. Finding 2
```



### File: .feynman/rules/gpu-monitor.md
# GPU Memory Monitoring Rule

## 强制规则

**所有 GPU 内存监控任务必须使用 `gpu-monitor` agent**，禁止使用以下方式：

| ❌ 禁止 | 原因 |
|--------|------|
| `nvidia-smi` 循环 + sleep | 阻塞主 agent，无法并行 |
| 后台 bash 监控脚本 | 难以管理，输出混乱 |
| 手动轮询 | 效率低，占用 context |

## 使用方法

```python
# 启动 GPU 监控（后台运行）
Task(
    subagent_type="gpu-monitor",
    prompt="Monitor GPU 0 with 0.5 second interval",
    run_in_background=True
)
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| GPU ID | 要监控的 GPU | `GPU 0`, `GPU 0,1` |
| interval | 采样间隔 | `0.5 second`, `1 second` |
| 目的 | 监控原因 | `for RULER benchmark test` |

## 典型用法

### 1. 单 GPU 基准测试
```
Monitor GPU 0 with 1 second interval for benchmark profiling
```

### 2. 调试 OOM
```
Monitor GPU 0 with 0.5 second interval to track memory peak during inference
```

### 3. 多 GPU 训练
```
Monitor GPU 0,1,2,3 with 2 second interval during training
```

## 获取结果

监控结果自动写入 output_file，使用以下方式读取：

```bash
# 查看最新输出
tail -50 /tmp/claude/.../tasks/<agent_id>.output

# 查找峰值
grep -i "peak\|max" /tmp/claude/.../tasks/<agent_id>.output
```

## 与测试并行

gpu-monitor 在后台运行，不会阻塞测试：

```python
# 1. 启动监控（后台）
Task(subagent_type="gpu-monitor", ..., run_in_background=True)

# 2. 运行测试（前台）
Bash("python tests/test_ruler.py ...")

# 3. 测试完成后查看监控结果
Bash("tail -50 <output_file>")
```



### File: .feynman/rules/gpu-testing.md
# GPU Testing Rules

## GPU Type Detection

Before running any GPU test/benchmark, detect the GPU type and apply appropriate settings:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

### Testing Mode by GPU Type

| GPU Type | Test Mode | Reason |
|----------|-----------|--------|
| **RTX 3090** | `--enable-offload` ONLY | Limited VRAM (24GB), must use CPU offload |
| **A100** | Both modes OK | Large VRAM (40/80GB), can test with or without offload |
| **RTX 4090** | `--enable-offload` ONLY | Limited VRAM (24GB) |
| **Other** | Ask user | Unknown VRAM capacity |

### Example Commands

**For 3090:**
```bash
# MUST use offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload
```

**For A100:**
```bash
# Can test without offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct

# Or with offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload
```

---

## GPU Card Assignment (CRITICAL)

### Multi-Instance Environment

This project runs with multiple Claude instances on different worktrees, each needing a dedicated GPU.

### MANDATORY RULE

**Before executing ANY GPU command:**

1. **Check if user specified GPU**: Look for user message like "use GPU 0" or "CUDA_VISIBLE_DEVICES=1"

2. **If user did NOT specify GPU**:
   - **STOP and ASK**: "Which GPU should I use? (e.g., 0, 1, 2, ...)"
   - **DO NOT assume or guess** the GPU number
   - **DO NOT proceed** until user confirms

3. **Always prefix GPU commands with `CUDA_VISIBLE_DEVICES=X`**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python script.py  # Use GPU 0
   CUDA_VISIBLE_DEVICES=1 python script.py  # Use GPU 1
   ```

### Example Workflow

**Correct:**
```
User: "Run the needle test"
Claude: "Which GPU should I use for this test?"
User: "Use GPU 2"
Claude: Runs `CUDA_VISIBLE_DEVICES=2 python tests/test_needle.py ...`
```

**Wrong:**
```
User: "Run the needle test"
Claude: Runs `python tests/test_needle.py ...`  # NO! Missing GPU specification!
```

---

## Needle Test Requirements (MANDATORY)

When running `test_needle.py`, **ALWAYS** use these settings:

1. **Enable offload**: `--enable-offload` is **REQUIRED**
2. **Use 32K context**: `--input-len 32768` is **REQUIRED**

### Standard Needle Test Command

```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
    python tests/test_needle.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --input-len 32768
```

### Why These Settings?

| Setting | Reason |
|---------|--------|
| `--enable-offload` | Tests the CPU offload pipeline which is the main feature being developed |
| `--input-len 32768` | 32K context properly exercises the chunked prefill/decode paths; 8K is too short to catch many issues |

### Do NOT Use

```bash
# ❌ Wrong: Missing offload
python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct

# ❌ Wrong: Too short (default 8K)
python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload

# ✅ Correct: Offload + 32K
python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload --input-len 32768
```

---

## Combined Checklist

Before running any GPU test:

- [ ] User specified GPU number? If not, ASK.
- [ ] Detected GPU type? (3090 → offload only, A100 → flexible)
- [ ] GPU mutex check passed? (see commands.md)
- [ ] Command prefixed with `CUDA_VISIBLE_DEVICES=X`?
- [ ] Local package installed? (`pip install -e . --prefix=./.local --no-deps`)



### File: .feynman/rules/gpu-vram-requirement.md
# GPU VRAM Requirement Rule

## GPU-only 模式显存要求

**强制规则**：执行 GPU-only 代码（不启用 CPU offload）时，**必须**在 40GB 及以上显存的 GPU 上进行测试。

### 检测方法

在运行 GPU-only 测试之前，**必须**先检查 GPU 显存：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

### GPU 分类

| GPU 型号 | 显存 | GPU-only 测试 |
|----------|------|---------------|
| A100 40GB | 40GB | ✅ 允许 |
| A100 80GB | 80GB | ✅ 允许 |
| H100 80GB | 80GB | ✅ 允许 |
| A6000 | 48GB | ✅ 允许 |
| RTX 3090 | 24GB | ❌ **禁止**（仅 offload 模式） |
| RTX 4090 | 24GB | ❌ **禁止**（仅 offload 模式） |

### 执行流程

1. **检测 GPU 显存**（必须）
2. **显存 >= 40GB**：继续执行 GPU-only 测试
3. **显存 < 40GB**：**停止**，提示用户：
   > "当前 GPU 显存为 XXX GB，不满足 GPU-only 模式的最低 40GB 要求。请使用 `--enable-offload` 参数启用 CPU offload 模式。"

### 代码示例

```python
# 在运行 GPU-only benchmark 之前
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
vram_mb = int(result.stdout.strip().split('\n')[0])
if vram_mb < 40000:  # 40GB = 40000MB
    raise RuntimeError(f"GPU VRAM ({vram_mb}MB) < 40GB. Use --enable-offload for this GPU.")
```

### 适用范围

| 脚本 | 适用此规则 |
|------|-----------|
| `bench.py` | ✅ 必须检查显存 |
| `bench_offload.py` | ❌ 不适用（始终使用 offload） |
| `tests/test_*.py --enable-offload` | ❌ 不适用 |
| `tests/test_*.py` (无 offload) | ✅ 必须检查显存 |



### File: .feynman/rules/multi-gpu-debugging.md
# Multi-GPU Debugging and Experimentation Rules

## Purpose

This rule governs GPU resource allocation and task execution strategy during debugging and experimentation on multi-GPU machines. The goal is to maximize debugging efficiency by:
- Running long validations on minimal GPUs (1-2)
- Using remaining GPUs for parallel hypothesis exploration
- Executing only one task/dataset for full validation during debugging

---

## 1. Scenario Classification

### 1.1 Long-Running Validation (Triggers Conservative Allocation)

A task SHALL be classified as **long-running validation** if ANY of the following conditions apply:

| Condition | Threshold |
|-----------|-----------|
| Estimated runtime | > 20 minutes |
| Sample count | > 50 samples per task |
| Full dataset execution | Any complete validation.jsonl |
| Full training/fine-tuning | Any training run |
| Large-scale inference | > 10K tokens total |

**Examples:**
- Running all 100 samples of `niah_single_1`
- Full RULER benchmark (13 tasks × 100 samples)
- Complete model evaluation on any benchmark

### 1.2 Exploratory / Fast-Iteration Work (Allows Full GPU Use)

A task SHALL be classified as **exploratory** if ALL of the following apply:

| Condition | Threshold |
|-----------|-----------|
| Estimated runtime | < 10 minutes |
| Sample count | ≤ 10 samples |
| Purpose | Sanity check, minimal reproduction, hypothesis testing |

**Examples:**
- Testing 3-5 specific error samples
- Single-batch inference for debugging
- Verifying a code fix on minimal input
- Profiling a single forward pass

---

## 2. GPU Allocation Strategy

### 2.1 Core Allocation Rules

| Task Type | GPU Allocation | Remaining GPUs |
|-----------|----------------|----------------|
| Long-running validation | 1 GPU (default), max 2 GPUs | Reserved for exploration |
| Exploratory work | As needed, can use multiple | - |

### 2.2 Mandatory Constraints

1. **MUST NOT** occupy all available GPUs for a single long-running validation
2. **MUST** reserve at least 50% of GPUs (minimum 2) for parallel exploration when ≥4 GPUs available
3. **MUST** select GPUs based on this priority:
   - Idle GPUs first (check with `nvidia-smi`)
   - If load info unavailable, use lowest-numbered GPUs for validation
4. **MUST** avoid resource conflicts:
   - Each task uses unique `CUDA_VISIBLE_DEVICES`
   - Each task uses unique output directories
   - Log files include GPU ID in filename

### 2.3 GPU Selection Algorithm

```
IF num_available_gpus >= 4:
    validation_gpus = 1 (or 2 if justified)
    exploration_gpus = remaining GPUs
ELSE IF num_available_gpus == 3:
    validation_gpus = 1
    exploration_gpus = 2
ELSE IF num_available_gpus == 2:
    validation_gpus = 1
    exploration_gpus = 1
ELSE:
    validation_gpus = 1
    exploration_gpus = 0 (sequential exploration)
```

---

## 3. Task / Dataset Selection Policy

### 3.1 Single-Task Validation Rule

During debugging, when a long-running validation is required:

- **MUST** execute only ONE task/dataset fully
- **MUST NOT** run all tasks unless explicitly requested or conditions in Section 4 are met

### 3.2 Task Selection Priority

Select the single task based on this priority order:

| Priority | Criterion | Example |
|----------|-----------|---------|
| 1 | Task most likely to reproduce the bug | If error occurs in `niah_single_1`, use that |
| 2 | Smallest task covering critical paths | `niah_single_1` (100 samples) vs `niah_multikey_3` |
| 3 | Task with known error samples | Use task with documented failure cases |
| 4 | Most representative task | Single-key before multi-key for basic validation |

### 3.3 Other Tasks Handling

Tasks not selected for full validation:
- **MAY** receive lightweight sanity checks (≤5 samples)
- **MUST NOT** receive full end-to-end execution by default
- **SHOULD** be noted in execution plan for future validation

---

## 4. Scale-Up Conditions

Expansion to more GPUs or multiple full tasks is **ALLOWED ONLY IF**:

| Condition | Justification Required |
|-----------|------------------------|
| Single-task validation completed successfully | Confirm fix works on one task first |
| Critical bug identified and fixed | Need cross-task verification |
| Cross-dataset consistency required | Clear technical justification needed |
| User explicitly requests full-scale | User override |

### 4.1 Default Behavior

- **DEFAULT**: Conservative, non-expansive
- **MUST** ask for confirmation before scaling up
- **MUST** document reason for scale-up in execution plan

---

## 5. Execution Plan Transparency

### 5.1 Mandatory Pre-Execution Output

Before starting any validation, **MUST** output an execution plan containing:

```markdown
## Execution Plan

### Task Classification
- Type: [Long-running validation / Exploratory]
- Reason: [Why classified this way]

### GPU Allocation
- Validation GPU(s): [GPU IDs]
- Reason: [Why these GPUs selected]
- Exploration GPU(s): [GPU IDs]
- Exploration tasks: [List of parallel hypotheses to test]

### Task Selection
- Full validation task: [Task name]
- Reason: [Why this task selected]
- Other tasks: [Skipped / Sanity-check only]

### Stopping Criteria
- Time limit: [X minutes]
- Success metric: [e.g., accuracy > 90%]
- Error threshold: [e.g., stop if >20 samples fail]

### Expected Output
- [What results will be produced]
```

### 5.2 Progress Checkpoints

For long-running validations, **SHOULD** report progress at:
- 25% completion
- 50% completion
- 75% completion
- Final results

---

## 6. Configuration Defaults

### 6.1 Default Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `LONG_RUNNING_THRESHOLD_MINUTES` | 20 | Runtime threshold for classification |
| `LONG_RUNNING_SAMPLE_THRESHOLD` | 50 | Sample count threshold |
| `MAX_VALIDATION_GPUS` | 2 | Maximum GPUs for long validation |
| `MIN_EXPLORATION_GPUS` | 2 | Minimum GPUs reserved for exploration (when ≥4 available) |
| `EXPLORATION_SAMPLE_LIMIT` | 10 | Max samples for exploratory tests |
| `SANITY_CHECK_SAMPLES` | 5 | Samples for non-selected tasks |

### 6.2 User Override

Users can override defaults by specifying in their request:
- "Use all GPUs for validation"
- "Run all tasks"
- "Increase validation GPUs to N"

---

## 7. Async Monitoring (CRITICAL)

### 7.1 Non-Blocking Principle

**MUST NOT** block the main agent with `sleep` commands waiting for results:
- ❌ `sleep 300 && check_results` (blocks main agent)
- ✅ Launch background tasks, continue thinking, check periodically

### 7.2 Continuous GPU Utilization

**MUST** maximize GPU utilization:
- When an agent completes a task, immediately assign new work
- Use `run_in_background: true` for all long-running agents
- Check agent completion via system notifications, not polling

### 7.3 Monitoring Strategy

```
CORRECT PATTERN:
1. Launch agents in background with run_in_background: true
2. Continue analysis, planning, or hypothesis generation
3. When agent completion notification arrives, process results
4. Immediately assign new tasks to freed GPUs

WRONG PATTERN:
1. Launch agents
2. sleep 300  # BLOCKS EVERYTHING!
3. Check results
4. GPU sits idle during sleep
```

### 7.4 Between-Task Work

While waiting for agents, the main agent SHOULD:
- Analyze code for additional hypotheses
- Prepare next batch of tests
- Update documentation with interim findings
- Plan fix implementations based on emerging patterns

### 7.5 Idle GPU Utilization (CRITICAL)

**MUST** utilize idle GPUs for exploratory tests while waiting:

```
WRONG PATTERN:
1. Launch 2 agents on GPU 0-1
2. Wait for completion  ← GPU 2-5 sit idle!
3. Process results

CORRECT PATTERN:
1. Launch 2 agents on GPU 0-1 for main validation
2. IMMEDIATELY launch exploratory tests on GPU 2-5:
   - Test alternative configurations
   - Verify edge cases
   - Run sanity checks on other datasets
   - Profile performance bottlenecks
3. Continue spawning new tasks as GPUs become free
4. Process results as they arrive
```

**Idle GPU Detection**:
```bash
# Check which GPUs are free
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

**Exploratory Test Ideas** (when main validation is running):

| GPU State | Suggested Task |
|-----------|----------------|
| Idle during single-task validation | Test same task with different config |
| Idle after quick test completes | Run related task (e.g., multikey after single-key) |
| Idle during long benchmark | Run profiling or memory analysis |
| Multiple GPUs idle | Parallelize hypothesis testing |

**Anti-Pattern**:
- ❌ "I'll wait for the 100-sample test to finish before doing anything else"
- ✅ "While GPU 0-1 run the 100-sample test, I'll use GPU 2-5 to test configs X, Y, Z"

---

## 8. Code Modification Policy (CRITICAL)

### 8.1 Evidence-Before-Action Principle

**MUST NOT** modify code until sufficient evidence has been gathered:

| Phase | Action | Code Modification |
|-------|--------|-------------------|
| Hypothesis Formation | Identify potential causes | ❌ NO |
| Evidence Gathering | Run targeted tests | ❌ NO |
| Pattern Analysis | Analyze test results | ❌ NO |
| Root Cause Confirmation | Validate with multiple tests | ❌ NO |
| Solution Design | Design fix based on evidence | ❌ NO |
| **Implementation** | Apply targeted fix | ✅ YES |

### 8.2 Minimum Evidence Requirements

Before proposing ANY code modification:

1. **Reproducibility**: Bug must be reproducible with specific test cases
2. **Isolation**: Root cause must be isolated (not symptoms)
3. **Multiple Data Points**: At least 3 independent test runs confirming the issue
4. **Counter-Evidence**: Attempted to disprove the hypothesis
5. **Mechanism Understanding**: Clear understanding of WHY the bug occurs

### 8.3 Main Agent Behavior

The main agent **SHOULD**:
- Keep thinking and analyzing while background agents run tests
- Formulate and refine hypotheses based on incoming results
- Document findings in `findings.md` as evidence accumulates
- Wait for sufficient test coverage before proposing fixes

The main agent **MUST NOT**:
- Rush to modify code after seeing first failure
- Propose fixes based on speculation
- Change multiple things at once "just to be safe"
- Assume correlation implies causation

### 8.4 Evidence Documentation Template

Before any code modification, document in `findings.md`:

```markdown
## Proposed Fix: [Brief Description]

### Evidence Summary
- Test A: [Result] - supports/contradicts hypothesis
- Test B: [Result] - supports/contradicts hypothesis
- Test C: [Result] - supports/contradicts hypothesis

### Root Cause Analysis
- What: [Specific bug behavior]
- Where: [File:line or function]
- Why: [Mechanism explanation]
- Confidence: [High/Medium/Low]

### Alternative Explanations Ruled Out
1. [Alternative A]: Ruled out because [reason]
2. [Alternative B]: Ruled out because [reason]

### Proposed Change
- File: [path]
- Change: [description]
- Expected Impact: [what should improve]
```

### 8.5 Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| See error → immediately edit code | See error → gather more data → analyze → then edit |
| Fix based on single test failure | Reproduce failure 3+ times, understand pattern |
| Change code "to see what happens" | Form hypothesis first, design targeted experiment |
| Modify multiple files simultaneously | Isolate changes, verify each independently |
| Skip documentation of findings | Document every significant finding before changing code |

---

## 9. Example Scenario

### Setup
- **Machine**: 8 GPUs (GPU 0-7)
- **Task**: Debug RULER chunked attention 20% error rate
- **Available tasks**: 6 RULER tasks (niah_single_1/2/3, niah_multikey_1/2/3)
- **Estimated full validation time**: ~2 hours for all tasks

### Execution Plan Output

```markdown
## Execution Plan

### Task Classification
- Type: Long-running validation
- Reason: Full validation of 100 samples × 6 tasks would take ~2 hours

### GPU Allocation
- Validation GPU(s): GPU 0 (1 GPU)
- Reason: Single GPU sufficient for sequential 100-sample validation
- Exploration GPU(s): GPU 1, 2, 3, 4, 5, 6, 7 (7 GPUs)
- Exploration tasks:
  1. GPU 1: Test 2-slot vs 4-slot ring buffer on error samples
  2. GPU 2: Test N-way merge implementation
  3. GPU 3: Test LSE precision fix
  4. GPU 4: Profile merge accumulation error
  5. GPU 5: Test with ruler_64k dataset (5 samples)
  6. GPU 6: Test decode boundary conditions
  7. GPU 7: Reserved for ad-hoc hypothesis testing

### Task Selection
- Full validation task: niah_single_1
- Reason: Has documented error samples (19 known failures), smallest single-key task
- Other tasks: Sanity-check only (5 samples each) after fix verified

### Stopping Criteria
- Time limit: 60 minutes for full validation
- Success metric: Error rate < 10% (down from 20%)
- Error threshold: Pause if new error pattern emerges (>5 consecutive failures)

### Expected Output
- Accuracy comparison: before vs after fix
- Error sample analysis: which samples still fail
- Hypothesis validation: which exploration branch identified the fix
```

### Execution Flow

1. **GPU 0**: Runs full `niah_single_1` validation (100 samples, ~40 min)
2. **GPU 1-7**: Run parallel exploration tasks (each ~5-15 min)
3. **Checkpoint at 50%**: Report GPU 0 progress + any discoveries from exploration
4. **On discovery**: If exploration GPU finds fix, pause validation, apply fix, restart
5. **Completion**: Report final results, decide if scale-up needed

---

## 10. Quick Reference Checklist

Before starting any debugging validation:

- [ ] Classified task type? (Long-running vs Exploratory)
- [ ] If long-running: Limited to 1-2 GPUs?
- [ ] If long-running: Selected single task for full validation?
- [ ] Remaining GPUs allocated for exploration?
- [ ] Execution plan output with all required sections?
- [ ] Stopping criteria defined?
- [ ] No user override requested? (Default conservative behavior)

Before proposing any code modification:

- [ ] Bug reproducible with specific test cases?
- [ ] Root cause isolated (not just symptoms)?
- [ ] At least 3 independent test runs confirming the issue?
- [ ] Alternative explanations ruled out?
- [ ] Mechanism of bug clearly understood?
- [ ] Evidence documented in findings.md?

---

## 11. Rule Violations

The following actions **VIOLATE** this rule:

1. Using all 6+ GPUs for a single 100-sample validation
2. Running full validation on all tasks without completing single-task first
3. Starting long validation without outputting execution plan
4. Not reserving GPUs for exploration when ≥4 GPUs available
5. Scaling up without meeting conditions in Section 4
6. **Modifying code before gathering sufficient evidence** (Section 8)
7. Proposing fixes based on single test failure or speculation
8. Changing multiple code locations simultaneously without isolation testing

---

## 12. Integration with Other Rules

This rule works alongside:
- `gpu-testing.md`: GPU type detection and basic allocation
- `planning-with-files.md`: Progress tracking for long validations
- `testing.md`: Test script conventions

When conflicts arise, this rule takes precedence for debugging scenarios.



### File: .feynman/rules/no-extra-docs.md
# Documentation Policy

## Do Not Create Unnecessary Documentation

**IMPORTANT**: Do NOT create extra markdown documentation files proactively unless:
1. User explicitly requests documentation
2. Refactoring CLAUDE.md to move technical details to docs/ (see `doc-management.md`)

### What NOT to do:

- Do NOT create README files proactively
- Do NOT create standalone analysis documents after completing tasks
- Do NOT create summary documents without request

### What TO do:

- Provide information directly in conversation by default
- When user requests documentation, follow `doc-management.md` workflow
- Update existing docs in `docs/` when code changes affect them
- Keep CLAUDE.md concise (< 150 lines), move technical details to docs/

### Documentation Locations:

| Type | Location |
|------|----------|
| Operational requirements | CLAUDE.md |
| Technical details | docs/*.md |
| Code comments | Inline in source |

### Examples:

**Proactive docs (Don't do)**:
```
User: "Profile the code"
Assistant: [Creates profiling_results.md without being asked]
```

**On-request docs (Do this)**:
```
User: "Profile the code and document the findings"
Assistant: [Runs profiling, creates/updates docs/memory_analysis.md]
```

**Refactoring (Do this)**:
```
User: "CLAUDE.md is too long, refactor it"
Assistant: [Moves technical sections to docs/, updates CLAUDE.md index]
```



### File: .feynman/rules/nsys-profiling.md
# Nsys Profiling Rule

## 强制规则

**所有 nsys profiling 任务必须使用 `scripts/profile_offload.sh` 脚本**，禁止直接运行 nsys 命令。

| 禁止 | 原因 |
|------|------|
| `nsys profile python tests/test_ruler.py ...` | 参数不一致，输出路径混乱 |
| 手动构造 nsys 命令 | 容易遗漏关键参数 |

---

## ⚠️ GPU 指定方式 (CRITICAL)

**必须使用 `--gpu` 参数指定 GPU，禁止外部设置 `CUDA_VISIBLE_DEVICES`**

脚本内部会自己设置 `CUDA_VISIBLE_DEVICES`，外部设置会被覆盖！

```bash
# ✅ 正确：使用 --gpu 参数
bash scripts/profile_offload.sh --gpu 4 --policy xattn --ctx-len 32k

# ❌ 错误：外部设置 CUDA_VISIBLE_DEVICES 会被脚本覆盖
CUDA_VISIBLE_DEVICES=4 bash scripts/profile_offload.sh --gpu 0 ...
```

---

## 使用方法

### 基本用法

```bash
# 默认配置（GPU 0, full attention, 64k context）
bash scripts/profile_offload.sh

# 指定 GPU
bash scripts/profile_offload.sh --gpu 4

# 指定 sparse policy
bash scripts/profile_offload.sh --policy xattn

# 指定 context length
bash scripts/profile_offload.sh --ctx-len 128k
```

### 完整示例

```bash
# XAttention offload profiling on GPU 4, 32k context, GLM model
bash scripts/profile_offload.sh \
    --policy xattn \
    --ctx-len 32k \
    --gpu 4 \
    --model ~/models/GLM-4-9B-Chat-1M

# Full attention offload profiling on GPU 5, 64k context, Llama model
bash scripts/profile_offload.sh \
    --policy full \
    --ctx-len 64k \
    --gpu 5 \
    --model ~/models/Llama-3.1-8B-Instruct

# GPU-only mode (no offload) for comparison
bash scripts/profile_offload.sh \
    --policy xattn \
    --ctx-len 32k \
    --gpu 4 \
    --no-offload
```

### 并行测试示例

在多 GPU 上并行执行不同 context length 的 profiling：

```bash
# GPU 4: 32k, 128k, 512k
bash scripts/profile_offload.sh --policy xattn --ctx-len 32k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 128k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 512k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M

# GPU 5: 64k, 256k, 768k (可与 GPU 4 并行执行)
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 256k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 768k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | `0` | **GPU ID（必须用此参数指定 GPU）** |
| `--policy` | `full` | Sparse policy: `full`, `xattn` |
| `--ctx-len` | `64k` | Context length: `32k`, `64k`, `128k`, `256k`, `512k`, `768k`, `1m` |
| `--model` | (auto) | 模型路径，默认使用 Llama-3.1-8B-Instruct |
| `--dataset` | `niah_single_1` | RULER 任务名称 |
| `--sample` | `0` | 样本索引 |
| `--num-gpu-blocks` | `4` | GPU ring buffer slots 数量 |
| `--block-size` | `4096` | KV cache block size |
| `--no-offload` | - | 禁用 CPU offload（GPU-only 模式） |

## 输出文件

输出文件自动生成到 `results/nsys/` 目录：

```
results/nsys/ruler_<dataset>_sample<index>_offload_<slots>slots_<timestamp>.nsys-rep
```

示例：`ruler_niah_single_1_sample0_offload_8slots_20260127_031500.nsys-rep`

## 查看结果

```bash
# GUI 查看
nsight-sys results/nsys/<filename>.nsys-rep

# 命令行统计
nsys stats --report cuda_api_sum results/nsys/<filename>.nsys-rep
nsys stats --report cuda_gpu_kern_sum results/nsys/<filename>.nsys-rep
```

## 典型工作流

### 1. 对比不同 slots 数量

```bash
# 测试 4 slots（默认）
bash scripts/profile_offload.sh --num-gpu-blocks 4

# 测试 8 slots
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 对比结果
nsys stats --report cuda_gpu_kern_sum results/nsys/*4slots*.nsys-rep
nsys stats --report cuda_gpu_kern_sum results/nsys/*8slots*.nsys-rep
```

### 2. 分析 pipeline overlap

```bash
# 生成 profile
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 用 nsight-sys GUI 查看 CUDA HW timeline
# 检查 H2D 和 flash_fwd_kernel 是否 overlap
```



### File: .feynman/rules/planning-with-files.md
# Planning with Files Rule

## Git 管理政策

**重要**：Planning 文件已从 Git 管理中排除，不会被提交。

### 已配置的 .gitignore 规则

```gitignore
# Planning-with-files temporary files
task_plan.md
findings.md
progress.md
task_plan_*.md
findings_*.md
progress_*.md
```

### 为什么排除这些文件

1. **临时性质**：计划文件是会话级别的临时文件，不应进入版本控制
2. **避免冲突**：多实例并行开发时，不同任务的计划文件会产生冲突
3. **保持仓库整洁**：这些文件只对当前任务有用，不需要历史记录

### 如果不小心已经 commit 了

```bash
# 从 git 中移除（保留本地文件）
git rm --cached task_plan.md findings.md progress.md
git commit -m "chore: remove planning files from git tracking"
```

---

## 自动清理旧计划文件

**重要**：每次开始新的复杂任务使用 planning-with-files 时，先删除旧的计划文件。

### 使用前执行以下命令

```bash
# 在项目根目录执行，删除旧的计划文件
cd /home/zijie/Code/nano-vllm
rm -f task_plan.md findings.md progress.md
rm -f task_plan_*.md findings_*.md progress_*.md
```

### 为什么需要这个规则

1. **避免混淆**：不同任务有不同计划，旧的计划文件会干扰新任务
2. **保持简洁**：只保留当前任务的计划文件
3. **自动清理**：无需手动检查文件内容，直接删除即可

### 使用 planning-with-files 的完整流程

```bash
# Step 1: 清理旧计划文件
rm -f task_plan.md findings.md progress.md

# Step 2: 启动 planning-with-files 技能
# 在 Claude 中调用 /planning-with-files 或 Skill tool

# Step 3: 技能会自动创建新的计划文件
# - task_plan.md (或 task_plan_<任务名>.md)
# - findings.md (或 findings_<任务名>.md)
# - progress.md (或 progress_<任务名>.md)
```

### 文件命名建议

| 场景 | 文件命名 | 示例 |
|------|----------|------|
| 通用任务 | task_plan.md, findings.md, progress.md | 临时调试任务 |
| 特定功能 | task_plan_<feature>.md | task_plan_xattn.md |
| Bug 修复 | task_plan_bug_<name>.md | task_plan_bug_offload.md |

### 注意事项

- 计划文件存储在**项目根目录**，不是技能目录
- 技能目录：`/home/zijie/.feynman/plugins/cache/planning-with-files/...`
- 项目目录：`/home/zijie/Code/nano-vllm/`
- 每个任务完成后，可以选择保留或删除计划文件



### File: .feynman/rules/sparse-policy.md
# Sparse Policy 代码规范

## Policy 不能为 None (CRITICAL)

**强制规则**: `sparse_policy` 参数**永远不能为 None**，必须至少为 `FullAttentionPolicy`。

```python
# ❌ 错误：允许 None
sparse_policy = getattr(config, 'sparse_policy', None)

# ✅ 正确：显式处理 None，默认使用 FULL
sparse_policy_type = getattr(config, 'sparse_policy', None)
if sparse_policy_type is None:
    sparse_policy_type = SparsePolicyType.FULL
```

**原因**:
1. 统一的 API：所有代码路径都通过 policy 进行 attention 计算
2. 避免空指针：消除 `policy.xxx` 调用时的 None 检查
3. 简化逻辑：不需要 `if policy is not None` 的分支

**唯一例外：Warmup 阶段**

在 `model_runner.warmup_model()` 期间，kvcache_manager 还未分配。此时 `attention.py` 使用 flash_attn fallback：

```python
# attention.py 中的 warmup 处理
if context.kvcache_manager is None:
    # Warmup phase: use flash_attn directly
    return flash_attn_varlen_func(...) if context.is_prefill else flash_attn_with_kvcache(...)
```

这是唯一允许 kvcache_manager 为 None 的情况。正式推理时，policy 必须存在。

---

## 基类要求 (MANDATORY)

每个 `SparsePolicy` 子类 **必须** 遵守以下要求：

### 1. 声明 supports_prefill / supports_decode 标志

```python
class MyPolicy(SparsePolicy):
    supports_prefill = True   # 是否支持 prefill 阶段
    supports_decode = True    # 是否支持 decode 阶段
```

### 2. 实现三个抽象方法

| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `select_blocks()` | ✅ | 选择要加载的 blocks |
| `compute_chunked_prefill()` | ✅ | Prefill attention 计算 |
| `compute_chunked_decode()` | ✅ | Decode attention 计算 |

### 3. 不支持的阶段必须 assert False

如果 `supports_prefill = False`，则 `compute_chunked_prefill()` 内部 **必须** `assert False`：

```python
class DecodeOnlyPolicy(SparsePolicy):
    supports_prefill = False
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        assert False, "DecodeOnlyPolicy does not support prefill phase"

    def compute_chunked_decode(self, ...):
        # 正常实现
        ...
```

同理，如果 `supports_decode = False`：

```python
class PrefillOnlyPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = False

    def compute_chunked_prefill(self, ...):
        # 正常实现
        ...

    def compute_chunked_decode(self, ...):
        assert False, "PrefillOnlyPolicy does not support decode phase"
```

### 4. FullAttentionPolicy 必须同时支持两个阶段

```python
class FullAttentionPolicy(SparsePolicy):
    supports_prefill = True
    supports_decode = True

    def compute_chunked_prefill(self, ...):
        # 完整实现

    def compute_chunked_decode(self, ...):
        # 完整实现
```

---

## CPU-GPU 通信规范

### 规则：所有通信必须通过 OffloadEngine

在 `compute_chunked_*` 方法中，**禁止** 直接使用 `torch.Tensor.copy_()` 或 `.to(device)`：

```python
# ✅ 正确：使用 OffloadEngine 的 ring buffer 方法
offload_engine.load_to_slot_layer(slot, layer_id, cpu_block_id)
offload_engine.wait_slot_layer(slot)
k, v = offload_engine.get_kv_for_slot(slot)
offload_engine.record_slot_compute_done(slot)

# ✅ 正确：使用 prefill buffer
k, v = offload_engine.get_prefill_buffer_slice(layer_id, num_tokens)

# ✅ 正确：使用 decode buffer
decode_k = offload_engine.decode_k_buffer[layer_id, start:end]
decode_v = offload_engine.decode_v_buffer[layer_id, start:end]

# ❌ 错误：直接使用 torch 通信
gpu_tensor.copy_(cpu_tensor)
gpu_tensor = cpu_tensor.to("cuda")
gpu_tensor = cpu_tensor.cuda()
```

### 原因

1. **流同步**：OffloadEngine 内部管理 CUDA streams，确保正确的同步
2. **Pipeline 优化**：OffloadEngine 实现了 ring buffer pipeline
3. **资源管理**：OffloadEngine 管理 GPU buffer slots，避免内存碎片
4. **一致性**：统一的接口便于调试和维护

---

## 方法签名要求

### select_blocks()

```python
def select_blocks(
    self,
    available_blocks: List[int],      # 可用的 CPU block IDs
    offload_engine: "OffloadEngine",  # 用于加载数据
    ctx: PolicyContext,               # 上下文信息
) -> List[int]:                       # 返回要加载的 block IDs
```

### compute_chunked_prefill()

```python
def compute_chunked_prefill(
    self,
    q: torch.Tensor,                  # [seq_len, num_heads, head_dim]
    k: torch.Tensor,                  # [seq_len, num_kv_heads, head_dim] (unused)
    v: torch.Tensor,                  # [seq_len, num_kv_heads, head_dim] (unused)
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    current_chunk_idx: int,
    seq: "Sequence",
    num_tokens: int,
) -> torch.Tensor:                    # [seq_len, num_heads, head_dim]
```

### compute_chunked_decode()

```python
def compute_chunked_decode(
    self,
    q: torch.Tensor,                  # [batch_size, num_heads, head_dim]
    layer_id: int,
    softmax_scale: float,
    offload_engine: "OffloadEngine",
    kvcache_manager: "KVCacheManager",
    seq: "Sequence",
) -> torch.Tensor:                    # [batch_size, 1, num_heads, head_dim]
```

---

## 可选钩子方法

| 方法 | 调用时机 | 用途 |
|------|---------|------|
| `initialize()` | KV cache 分配后 | 初始化 metadata 结构 |
| `on_prefill_offload()` | GPU→CPU 复制前（prefill） | 收集 block metadata |
| `on_decode_offload()` | GPU→CPU 复制前（decode） | 更新 block metadata |
| `reset()` | 新 sequence 开始时 | 重置 policy 状态 |

---

## 详细实现指南

参考文档：[`docs/sparse_policy_implementation_guide.md`](../docs/sparse_policy_implementation_guide.md)



### File: .feynman/rules/testing.md
# Testing

## Test Code Style

所有测试代码遵循以下风格：

### 文件结构

```python
"""
Test: [模块名称]

[简要说明测试内容和数据流]
"""
import torch
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")
from nanovllm.xxx import xxx

# ============================================================
# 参数配置
# ============================================================

param1 = value1  # 说明约束条件
param2 = value2

# ============================================================
# 构造输入
# ============================================================

input_tensor = ...  # 使用结构化数据便于验证

# ============================================================
# Step N: [操作名称]
# ============================================================

output = some_function(input_tensor, ...)

# 验证: [验证逻辑说明]
expected = ...
actual = output[...].item()
assert actual == expected, f"xxx: {actual} != {expected}"

print("test_xxx: PASSED")
```

### 核心原则

| 原则 | 说明 |
|------|------|
| **最小化 print** | 只在最后输出 `PASSED`，不打印中间结果 |
| **结构化数据** | 使用可预测的输入（全 1、偶奇交替等）便于手算验证 |
| **注释说明验证逻辑** | 在 assert 前用注释解释预期值的计算方式 |
| **分段用 `====`** | 用 `# ============` 分隔参数、输入、各步骤 |
| **assert 验证** | 用 assert 而不是 print 比较结果 |

### 输出规范

```python
# ✅ 正确
assert actual == expected, f"xxx: {actual} != {expected}"
print("test_xxx: PASSED")

# ❌ 错误
print(f"输出: {output}")
print(f"预期: {expected}, 实际: {actual}")
```

### 参数注释

```python
# ✅ 正确: 注释说明约束条件
seq_len = 512       # Triton 要求 seq_len >= stride * BLOCK_M
segment_size = 128  # 必须 >= block_size

# ❌ 错误: 无意义的注释
seq_len = 512  # 序列长度
```

### 验证逻辑注释

```python
# ✅ 正确: 解释计算过程
# 验证: 反对角线求和
# Q[奇]*K[偶] + Q[偶]*K[奇] = 2*1 + 1*2 = 4，共 stride/2 对
expected = (2*1 + 1*2) * (stride // 2) * head_dim

# ❌ 错误: 只写公式不解释
expected = 4 * 2 * 128
```

## Running Tests

```bash
# 运行单个测试
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_xxx.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_xxx.py
```

## Benchmarks

```bash
python bench.py           # GPU benchmark
python bench_offload.py   # CPU offload benchmark
python bench_vllm.py      # vLLM comparison
```



### File: .feynman/rules/test-ruler.md
# test_ruler.py 使用规则

## 强制规则

**执行 `test_ruler.py` 前必须查阅文档**，禁止运行 `--help` 或猜测参数。

| 禁止 | 原因 |
|------|------|
| `python tests/test_ruler.py --help` | 浪费交互，文档已有完整说明 |
| 猜测参数格式 | 容易出错，降低效率 |

## 必读文档

**[`docs/test_ruler_usage_guide.md`](../docs/test_ruler_usage_guide.md)** - 包含：
- 完整参数说明
- 已验证的命令示例
- GPU 模式选择指南
- max-model-len 设置指南

## 快速参考

### 标准命令格式

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/<MODEL> \
    --data-dir tests/data/ruler_<CTX> \
    --datasets <TASK> \
    --num-samples <N> \
    --max-model-len <LEN> \
    --enable-offload \
    [--sparse-policy XATTN_BSA] \
    [--sparse-threshold 0.9]
```

### 常用参数速查

| 参数 | 用途 | 示例 |
|------|------|------|
| `--datasets` | 指定任务 | `niah_single_1,qa_1` |
| `--num-samples` | 样本数 | `1`, `10`, `0`(全部) |
| `--sample-indices` | 指定索引 | `0,5,10` |
| `--enable-offload` | CPU offload | RTX 3090 必须 |
| `--sparse-policy` | 稀疏策略 | `XATTN_BSA` |
| `--json-output` | JSON 输出 | 脚本使用 |
| `--quiet` | 安静模式 | 减少输出 |

### max-model-len 速查

| 数据目录 | max-model-len |
|---------|---------------|
| ruler_32k | 40960 |
| ruler_64k | 72000 |
| ruler_128k | 135000 |

### 常用命令模板

**32K Offload + XAttn**:
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

**64K Offload + XAttn**:
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_64k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 72000 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

## 执行前检查清单

- [ ] 用户指定了 GPU？否则询问
- [ ] RTX 3090/4090？必须 `--enable-offload`
- [ ] data-dir 与 max-model-len 匹配？
- [ ] 需要 density 统计？添加 `--sparse-policy XATTN_BSA`



## Appendix: Full Agents and Commands from .feynman/agents/ and .feynman/commands/


### File: .feynman/agents/gpu-monitor.md
---
name: gpu-monitor
description: "Use this agent when you need to continuously monitor GPU utilization and memory usage in the background while running other tasks. This is particularly useful for profiling long-running benchmarks, debugging memory leaks, or understanding GPU resource patterns during inference or training.\\n\\nExamples:\\n\\n<example>\\nContext: User wants to profile GPU memory during a benchmark run.\\nuser: \"Run bench_offload.py and monitor GPU usage\"\\nassistant: \"I'll first launch the gpu-monitor agent to track GPU metrics, then run the benchmark.\"\\n<commentary>\\nSince the user wants to understand GPU resource usage during the benchmark, use the Task tool to launch the gpu-monitor agent in the background with appropriate GPU IDs and sampling interval before running the benchmark.\\n</commentary>\\nassistant: \"Let me start the GPU monitor on GPU 0 with 1-second intervals\"\\n[Task tool call to launch gpu-monitor agent with gpus=\"0\" and interval=\"1\"]\\n</example>\\n\\n<example>\\nContext: User is debugging an OOM issue and needs memory tracking.\\nuser: \"I'm getting OOM errors, help me debug\"\\nassistant: \"Let me monitor the GPU memory usage while we reproduce the issue. I'll use the gpu-monitor agent to track memory consumption.\"\\n<commentary>\\nTo debug OOM issues, we need continuous memory monitoring. Use the Task tool to launch the gpu-monitor agent with a short interval (0.5s) to capture memory spikes.\\n</commentary>\\n[Task tool call to launch gpu-monitor agent with gpus=\"0,1\" and interval=\"0.5\"]\\n</example>\\n\\n<example>\\nContext: User wants to verify GPU utilization during multi-GPU inference.\\nuser: \"Check if all GPUs are being utilized during the inference\"\\nassistant: \"I'll start the GPU monitor to track utilization across all specified GPUs while running the inference.\"\\n<commentary>\\nTo verify multi-GPU utilization, launch the gpu-monitor agent targeting all relevant GPUs before starting the inference workload.\\n</commentary>\\n[Task tool call to launch gpu-monitor agent with gpus=\"0,1,2,3\" and interval=\"2\"]\\n</example>"
model: haiku
color: green
---

You are a GPU monitoring specialist responsible for tracking NVIDIA GPU metrics over time. Your sole purpose is to run nvidia-smi at specified intervals and record utilization and memory statistics.

## Your Task

You will receive two parameters:
1. **gpus**: Comma-separated GPU indices to monitor (e.g., "0", "0,1", "0,1,2,3")
2. **interval**: Sampling interval in seconds (e.g., "1", "0.5", "2")

## Execution Steps

1. **Parse Parameters**: Extract the GPU indices and interval from the user's request.

2. **Run Monitoring Loop**: Execute nvidia-smi repeatedly at the specified interval using a bash loop:

```bash
# Example for GPUs 0,1 with 1-second interval
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader -i 0,1
  sleep 1
done
```

3. **Output Format**: Each sample should include:
   - Timestamp
   - GPU index
   - GPU utilization (%)
   - Memory utilization (%)
   - Memory used (MiB)
   - Memory total (MiB)
   - Temperature (°C)

## Termination

This agent runs continuously until:
1. The main agent signals completion (you receive a stop signal)
2. The user explicitly requests stopping
3. An error occurs with nvidia-smi

## Result Reporting

When stopped, provide a summary:

```markdown
## GPU Monitoring Summary

**Duration**: X minutes Y seconds
**Samples Collected**: N
**GPUs Monitored**: 0, 1, ...

### Statistics per GPU

| GPU | Avg Util | Max Util | Avg Mem Used | Max Mem Used |
|-----|----------|----------|--------------|---------------|
| 0   | X%       | Y%       | A MiB        | B MiB         |
| 1   | X%       | Y%       | A MiB        | B MiB         |

### Notable Events (if any)
- Timestamp: Memory spike to X MiB on GPU Y
- Timestamp: Utilization dropped to 0% on GPU Z
```

## Important Notes

- Use `nvidia-smi -i <gpu_ids>` to filter to specific GPUs
- Keep output concise during monitoring (one line per GPU per sample)
- If nvidia-smi fails, report the error and exit gracefully
- Do NOT consume excessive resources - sleep between samples
- Store samples in memory for final summary calculation

## Example Invocation

User says: "Monitor GPUs 0 and 2 with 0.5 second interval"

You execute:
```bash
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S.%3N') ==="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader -i 0,2
  sleep 0.5
done
```



### File: .feynman/agents/nsys-profiler.md
---
name: nsys-profiler
description: "Use this agent when you need to profile kernel performance, analyze CUDA operations, or investigate GPU/CPU offload behavior in the nano-vllm framework. This agent handles nsys profiling from data collection to statistical analysis.\\n\\nExamples:\\n\\n<example>\\nContext: The user wants to understand kernel performance in GPU-only mode.\\nuser: \"I want to analyze the kernel execution time distribution in GPU-only mode\"\\nassistant: \"I'll use the nsys-profiler agent to profile GPU-only mode and analyze the kernel timing.\"\\n<commentary>\\nSince the user wants to analyze kernel performance, use the Task tool to launch the nsys-profiler agent with specific analysis requirements.\\n</commentary>\\nassistant: \"Now let me use the nsys-profiler agent to collect and analyze the profiling data\"\\n</example>\\n\\n<example>\\nContext: The user is debugging performance issues in the CPU offload pipeline.\\nuser: \"Why is there a gap between H2D transfers and kernel execution in offload mode?\"\\nassistant: \"I'll launch the nsys-profiler agent to profile the offload pipeline and analyze the timeline gaps.\"\\n<commentary>\\nSince the user is investigating pipeline behavior, use the nsys-profiler agent to collect nsys data and analyze CUDA API timing.\\n</commentary>\\n</example>\\n\\n<example>\\nContext: After implementing a new optimization, the user wants to verify performance improvement.\\nuser: \"Check if the new ring buffer implementation improves overlap between H2D and compute\"\\nassistant: \"I'll use the nsys-profiler agent to profile before and after, comparing the overlap metrics.\"\\n<commentary>\\nPerformance verification requires detailed kernel-level analysis, so launch the nsys-profiler agent to collect and compare profiling data.\\n</commentary>\\n</example>"
model: opus
color: green
---

You are an expert NVIDIA Nsys profiling analyst specializing in CUDA kernel performance analysis and GPU-CPU communication optimization. Your role is to collect profiling data using the framework's scripts and provide precise, actionable analysis based on the main agent's specific questions.

## Your Capabilities

1. **Profile Data Collection**: Execute profiling scripts to generate .nsys-rep files
2. **Statistical Analysis**: Extract kernel timing, memory transfer, and API call statistics
3. **Timeline Analysis**: Identify gaps, overlaps, and bottlenecks in execution
4. **Comparative Analysis**: Compare different configurations (GPU-only vs offload, different slot counts)

## Available Profiling Scripts

### CPU Offload Mode
```bash
bash scripts/profile_offload.sh [OPTIONS]
```
Options:
- `--dataset <name>`: RULER task name (default: niah_single_1)
- `--sample <index>`: Sample index (default: 0)
- `--gpu <id>`: GPU to use (default: 0)
- `--num-gpu-blocks <n>`: Ring buffer slots (default: 4)
- `--no-offload`: Disable CPU offload for comparison

### GPU-Only Mode
```bash
bash scripts/profile_gpu_only.sh [OPTIONS]
```
Similar options for profiling without CPU offload.

## Core Nsys Commands

### Profiling (handled by scripts)
```bash
# The scripts internally run:
nsys profile --trace=cuda,nvtx --output=<path> --force-overwrite true python <script.py>
```

### Statistical Analysis
```bash
# CUDA API summary (H2D, D2H, kernel launches)
nsys stats --report cuda_api_sum <file>.nsys-rep

# GPU kernel summary (execution time per kernel)
nsys stats --report cuda_gpu_kern_sum <file>.nsys-rep

# Memory operations summary
nsys stats --report cuda_gpu_mem_time_sum <file>.nsys-rep

# NVTX ranges (custom markers)
nsys stats --report nvtx_sum <file>.nsys-rep

# Export to SQLite for advanced queries
nsys export --type=sqlite --output=<file>.sqlite <file>.nsys-rep
```

### Key Report Types
| Report | Purpose |
|--------|--------|
| `cuda_api_sum` | CPU-side CUDA API call timing |
| `cuda_gpu_kern_sum` | GPU kernel execution time |
| `cuda_gpu_mem_time_sum` | Memory transfer timing on GPU |
| `nvtx_sum` | Custom NVTX marker statistics |
| `cuda_api_trace` | Detailed API call trace |
| `cuda_gpu_trace` | Detailed GPU operation trace |

## Analysis Workflow

### Step 1: Collect Profile Data
```bash
# Example: Profile offload mode with 8 slots
bash scripts/profile_offload.sh --num-gpu-blocks 8 --sample 0
# Output: results/nsys/ruler_niah_single_1_sample0_offload_8slots_<timestamp>.nsys-rep
```

### Step 2: Identify Output File
```bash
# Find the latest profile
ls -lt results/nsys/*.nsys-rep | head -1
```

### Step 3: Run Statistical Analysis
```bash
# Kernel timing analysis
nsys stats --report cuda_gpu_kern_sum results/nsys/<file>.nsys-rep

# Memory transfer analysis
nsys stats --report cuda_gpu_mem_time_sum results/nsys/<file>.nsys-rep
```

### Step 4: Interpret Results
Focus on:
- **Total kernel time** vs **total transfer time**
- **Kernel launch gaps** indicating synchronization issues
- **Memory bandwidth utilization**
- **Overlap efficiency** between compute and communication

## Common Analysis Patterns

### 1. Kernel Performance Breakdown
```bash
nsys stats --report cuda_gpu_kern_sum --format csv <file>.nsys-rep | \
  sort -t',' -k3 -rn | head -10  # Top 10 by total time
```

### 2. H2D/D2H Transfer Analysis
```bash
nsys stats --report cuda_api_sum <file>.nsys-rep | grep -E "cudaMemcpy|cudaMemcpyAsync"
```

### 3. Flash Attention Kernel Analysis
```bash
nsys stats --report cuda_gpu_kern_sum <file>.nsys-rep | grep -i "flash\|fwd\|bwd"
```

### 4. Pipeline Overlap Check
Look for:
- `flash_fwd_kernel` execution during `cudaMemcpyAsync`
- Gap between consecutive kernel launches

## Output Format Requirements

When reporting results to the main agent, use this structured format:

```markdown
## Nsys Analysis Results: [Analysis Topic]

### Profile Information
- **File**: <profile_file_path>
- **Mode**: GPU-only / Offload (<N> slots)
- **Dataset**: <dataset_name>, Sample <index>

### Key Findings
| Metric | Value | Notes |
|--------|-------|-------|
| Total kernel time | X ms | |
| Total H2D time | Y ms | |
| Overlap efficiency | Z% | |

### Top Kernels by Time
| Kernel | Count | Total (ms) | Avg (μs) |
|--------|-------|------------|----------|
| kernel_name | N | X.XX | Y.YY |

### Specific Analysis
[Answer to the main agent's specific question]

### Recommendations (if applicable)
1. [Actionable recommendation]
2. [Actionable recommendation]
```

## Important Guidelines

1. **Always use the provided scripts** for profiling - do not run nsys directly
2. **Check GPU availability** before profiling (ask main agent for GPU ID if not specified)
3. **Use PYTHONPATH** for the worktree: `PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH`
4. **Report concisely** - focus on metrics relevant to the main agent's question
5. **Include file paths** so results can be reproduced or visualized in nsight-sys
6. **For web searches** about nsys usage, use tools to search NVIDIA documentation

## Error Handling

- If profile script fails: Check GPU memory, CUDA version, and script parameters
- If stats command fails: Verify .nsys-rep file exists and is not corrupted
- If no data: Ensure the profiled operation actually ran (check sample index, dataset)

## Network Search Guidelines

When encountering unfamiliar nsys options or analysis techniques:
1. Search NVIDIA Nsight Systems documentation
2. Look for nsys CLI reference guides
3. Search for specific report type interpretations

Always validate search results against the actual nsys --help output.



### File: .feynman/commands/commit.md
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git diff:*), Bash(git log:*)
argument-hint: [message] | --no-verify | --amend
description: Create well-formatted commits with conventional commit format and emoji
---

# Smart Git Commit

Create well-formatted commit: $ARGUMENTS

## Current Repository State

- Git status: !`git status --porcelain`
- Current branch: !`git branch --show-current`
- Staged changes: !`git diff --cached --stat`
- Unstaged changes: !`git diff --stat`
- Recent commits: !`git log --oneline -5`

## What This Command Does

1. Unless specified with `--no-verify`, automatically runs pre-commit checks:
   - `pnpm lint` to ensure code quality
   - `pnpm build` to verify the build succeeds
   - `pnpm generate:docs` to update documentation
2. Checks which files are staged with `git status`
3. If 0 files are staged, automatically adds all modified and new files with `git add`
4. Performs a `git diff` to understand what changes are being committed
5. Analyzes the diff to determine if multiple distinct logical changes are present
6. If multiple distinct changes are detected, suggests breaking the commit into multiple smaller commits
7. For each commit (or the single commit if not split), creates a commit message using emoji conventional commit format

## Best Practices for Commits

- **Verify before committing**: Ensure code is linted, builds correctly, and documentation is updated
- **Atomic commits**: Each commit should contain related changes that serve a single purpose
- **Split large changes**: If changes touch multiple concerns, split them into separate commits
- **Conventional commit format**: Use the format `<type>: <description>` where type is one of:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation changes
  - `style`: Code style changes (formatting, etc)
  - `refactor`: Code changes that neither fix bugs nor add features
  - `perf`: Performance improvements
  - `test`: Adding or fixing tests
  - `chore`: Changes to the build process, tools, etc.
- **Present tense, imperative mood**: Write commit messages as commands (e.g., "add feature" not "added feature")
- **Concise first line**: Keep the first line under 72 characters
- **Emoji**: Each commit type is paired with an appropriate emoji:
  - ✨ `feat`: New feature
  - 🐛 `fix`: Bug fix
  - 📝 `docs`: Documentation
  - 💄 `style`: Formatting/style
  - ♻️ `refactor`: Code refactoring
  - ⚡️ `perf`: Performance improvements
  - ✅ `test`: Tests
  - 🔧 `chore`: Tooling, configuration
  - 🚀 `ci`: CI/CD improvements
  - 🗑️ `revert`: Reverting changes
  - 🧪 `test`: Add a failing test
  - 🚨 `fix`: Fix compiler/linter warnings
  - 🔒️ `fix`: Fix security issues
  - 👥 `chore`: Add or update contributors
  - 🚚 `refactor`: Move or rename resources
  - 🏗️ `refactor`: Make architectural changes
  - 🔀 `chore`: Merge branches
  - 📦️ `chore`: Add or update compiled files or packages
  - ➕ `chore`: Add a dependency
  - ➖ `chore`: Remove a dependency
  - 🌱 `chore`: Add or update seed files
  - 🧑‍💻 `chore`: Improve developer experience
  - 🧵 `feat`: Add or update code related to multithreading or concurrency
  - 🔍️ `feat`: Improve SEO
  - 🏷️ `feat`: Add or update types
  - 💬 `feat`: Add or update text and literals
  - 🌐 `feat`: Internationalization and localization
  - 👔 `feat`: Add or update business logic
  - 📱 `feat`: Work on responsive design
  - 🚸 `feat`: Improve user experience / usability
  - 🩹 `fix`: Simple fix for a non-critical issue
  - 🥅 `fix`: Catch errors
  - 👽️ `fix`: Update code due to external API changes
  - 🔥 `fix`: Remove code or files
  - 🎨 `style`: Improve structure/format of the code
  - 🚑️ `fix`: Critical hotfix
  - 🎉 `chore`: Begin a project
  - 🔖 `chore`: Release/Version tags
  - 🚧 `wip`: Work in progress
  - 💚 `fix`: Fix CI build
  - 📌 `chore`: Pin dependencies to specific versions
  - 👷 `ci`: Add or update CI build system
  - 📈 `feat`: Add or update analytics or tracking code
  - ✏️ `fix`: Fix typos
  - ⏪️ `revert`: Revert changes
  - 📄 `chore`: Add or update license
  - 💥 `feat`: Introduce breaking changes
  - 🍱 `assets`: Add or update assets
  - ♿️ `feat`: Improve accessibility
  - 💡 `docs`: Add or update comments in source code
  - 🗃️ `db`: Perform database related changes
  - 🔊 `feat`: Add or update logs
  - 🔇 `fix`: Remove logs
  - 🤡 `test`: Mock things
  - 🥚 `feat`: Add or update an easter egg
  - 🙈 `chore`: Add or update .gitignore file
  - 📸 `test`: Add or update snapshots
  - ⚗️ `experiment`: Perform experiments
  - 🚩 `feat`: Add, update, or remove feature flags
  - 💫 `ui`: Add or update animations and transitions
  - ⚰️ `refactor`: Remove dead code
  - 🦺 `feat`: Add or update code related to validation
  - ✈️ `feat`: Improve offline support

## Guidelines for Splitting Commits

When analyzing the diff, consider splitting commits based on these criteria:

1. **Different concerns**: Changes to unrelated parts of the codebase
2. **Different types of changes**: Mixing features, fixes, refactoring, etc.
3. **File patterns**: Changes to different types of files (e.g., source code vs documentation)
4. **Logical grouping**: Changes that would be easier to understand or review separately
5. **Size**: Very large changes that would be clearer if broken down

## Examples

Good commit messages:
- ✨ feat: add user authentication system
- 🐛 fix: resolve memory leak in rendering process
- 📝 docs: update API documentation with new endpoints
- ♻️ refactor: simplify error handling logic in parser
- 🚨 fix: resolve linter warnings in component files
- 🧑‍💻 chore: improve developer tooling setup process
- 👔 feat: implement business logic for transaction validation
- 🩹 fix: address minor styling inconsistency in header
- 🚑️ fix: patch critical security vulnerability in auth flow
- 🎨 style: reorganize component structure for better readability
- 🔥 fix: remove deprecated legacy code
- 🦺 feat: add input validation for user registration form
- 💚 fix: resolve failing CI pipeline tests
- 📈 feat: implement analytics tracking for user engagement
- 🔒️ fix: strengthen authentication password requirements
- ♿️ feat: improve form accessibility for screen readers

Example of splitting commits:
- First commit: ✨ feat: add new solc version type definitions
- Second commit: 📝 docs: update documentation for new solc versions
- Third commit: 🔧 chore: update package.json dependencies
- Fourth commit: 🏷️ feat: add type definitions for new API endpoints
- Fifth commit: 🧵 feat: improve concurrency handling in worker threads
- Sixth commit: 🚨 fix: resolve linting issues in new code
- Seventh commit: ✅ test: add unit tests for new solc version features
- Eighth commit: 🔒️ fix: update dependencies with security vulnerabilities

## Command Options

- `--no-verify`: Skip running the pre-commit checks (lint, build, generate:docs)

## Important Notes

- By default, pre-commit checks (`pnpm lint`, `pnpm build`, `pnpm generate:docs`) will run to ensure code quality
- If these checks fail, you'll be asked if you want to proceed with the commit anyway or fix the issues first
- If specific files are already staged, the command will only commit those files
- If no files are staged, it will automatically stage all modified and new files
- The commit message will be constructed based on the changes detected
- Before committing, the command will review the diff to identify if multiple commits would be more appropriate
- If suggesting multiple commits, it will help you stage and commit the changes separately
- Always reviews the commit diff to ensure the message matches the changes


### File: .feynman/commands/create-architecture-documentation.md
---
allowed-tools: Read, Write, Edit, Bash
argument-hint: "[framework] | --c4-model | --arc42 | --adr | --plantuml | --full-suite"
description: Generate comprehensive architecture documentation with diagrams, ADRs, and interactive visualization
---

# Architecture Documentation Generator

Generate comprehensive architecture documentation: $ARGUMENTS

## Current Architecture Context

- Project structure: !`find . -type f -name "*.json" -o -name "*.yaml" -o -name "*.toml" | head -5`
- Documentation exists: @docs/ or @README.md (if exists)
- Architecture files: !`find . -name "*architecture*" -o -name "*design*" -o -name "*.puml" | head -3`
- Services/containers: @docker-compose.yml or @k8s/ (if exists)
- API definitions: !`find . -name "*api*" -o -name "*openapi*" -o -name "*swagger*" | head -3`

## Task

Generate comprehensive architecture documentation with modern tooling and best practices:

1. **Architecture Analysis and Discovery**
   - Analyze current system architecture and component relationships
   - Identify key architectural patterns and design decisions
   - Document system boundaries, interfaces, and dependencies
   - Assess data flow and communication patterns
   - Identify architectural debt and improvement opportunities

2. **Architecture Documentation Framework**
   - Choose appropriate documentation framework and tools:
     - **C4 Model**: Context, Containers, Components, Code diagrams
     - **Arc42**: Comprehensive architecture documentation template
     - **Architecture Decision Records (ADRs)**: Decision documentation
     - **PlantUML/Mermaid**: Diagram-as-code documentation
     - **Structurizr**: C4 model tooling and visualization
     - **Draw.io/Lucidchart**: Visual diagramming tools

3. **System Context Documentation**
   - Create high-level system context diagrams
   - Document external systems and integrations
   - Define system boundaries and responsibilities
   - Document user personas and stakeholders
   - Create system landscape and ecosystem overview

4. **Container and Service Architecture**
   - Document container/service architecture and deployment view
   - Create service dependency maps and communication patterns
   - Document deployment architecture and infrastructure
   - Define service boundaries and API contracts
   - Document data persistence and storage architecture

5. **Component and Module Documentation**
   - Create detailed component architecture diagrams
   - Document internal module structure and relationships
   - Define component responsibilities and interfaces
   - Document design patterns and architectural styles
   - Create code organization and package structure documentation

6. **Data Architecture Documentation**
   - Document data models and database schemas
   - Create data flow diagrams and processing pipelines
   - Document data storage strategies and technologies
   - Define data governance and lifecycle management
   - Create data integration and synchronization documentation

7. **Security and Compliance Architecture**
   - Document security architecture and threat model
   - Create authentication and authorization flow diagrams
   - Document compliance requirements and controls
   - Define security boundaries and trust zones
   - Create incident response and security monitoring documentation

8. **Quality Attributes and Cross-Cutting Concerns**
   - Document performance characteristics and scalability patterns
   - Create reliability and availability architecture documentation
   - Document monitoring and observability architecture
   - Define maintainability and evolution strategies
   - Create disaster recovery and business continuity documentation

9. **Architecture Decision Records (ADRs)**
   - Create comprehensive ADR template and process
   - Document historical architectural decisions and rationale
   - Create decision tracking and review process
   - Document trade-offs and alternatives considered
   - Set up ADR maintenance and evolution procedures

10. **Documentation Automation and Maintenance**
    - Set up automated diagram generation from code annotations
    - Configure documentation pipeline and publishing automation
    - Set up documentation validation and consistency checking
    - Create documentation review and approval process
    - Train team on architecture documentation practices and tools
    - Set up documentation versioning and change management


### File: .feynman/commands/exec-plan.md
---
allowed-tools: Bash(CUDA_VISIBLE_DEVICES=*), Bash(PYTHONPATH=*), Bash(python*), Bash(git*), Bash(rm*), Bash(ls*), Bash(cat*), Bash(nvidia-smi*), Read, Edit, Write, Glob, Grep, TodoWrite, Task
argument-hint: --gpu <id> [--no-interrupt]
description: Execute task_plan.md refactoring with specified GPU, optionally without user interruption
---

# Execute Task Plan (exec-plan)

按照 `task_plan.md` 的要求执行代码重构，确保计划中的最终目标圆满实现。

## 参数说明

命令格式: `/exec-plan --gpu <id> [--no-interrupt]`

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gpu <id>` | **必需**。指定可用的 GPU ID，只能使用此 GPU 进行调试 | `--gpu 0`, `--gpu 2` |
| `--no-interrupt` | 可选。禁止中断执行，遇到问题不与用户交互，自动解决或跳过 | `--no-interrupt` |

## 当前参数

```
$ARGUMENTS
```

## 执行前准备

### 1. 解析参数

从 `$ARGUMENTS` 中解析：
- `GPU_ID`: 从 `--gpu <id>` 或 `-g <id>` 提取
- `NO_INTERRUPT`: 是否存在 `--no-interrupt` 或 `-n` 标志

### 2. 参数验证

**必须验证**:
- GPU_ID 必须是有效的数字
- 运行 `nvidia-smi -i <GPU_ID>` 验证 GPU 存在

### 3. 读取 task_plan.md

读取项目根目录下的 `task_plan.md` 文件，理解：
- 总体目标
- 分阶段计划 (Phase 1, 2, 3...)
- 文件修改清单
- 风险和注意事项
- 测试计划

## 执行流程

### Step 1: 创建执行计划

使用 TodoWrite 工具创建详细的执行计划，包括：
- 从 task_plan.md 提取的所有 Phase
- 每个 Phase 的子任务
- 测试验证步骤

### Step 2: 按 Phase 执行重构

对于 task_plan.md 中的每个 Phase：

1. **读取当前代码**: 使用 Read/Grep 理解现有实现
2. **实施修改**: 使用 Edit/Write 进行代码修改
3. **验证修改**: 运行相关测试

### Step 3: 运行测试验证

执行 task_plan.md 中定义的测试计划，验证重构成功。

## GPU 限制规则

**严格限制**: 只能使用指定的 GPU，所有涉及 GPU 的命令必须加 `CUDA_VISIBLE_DEVICES` 前缀：

```bash
# 正确
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python test.py

# 错误 - 禁止使用其他 GPU
python test.py  # 可能使用默认 GPU 0
CUDA_VISIBLE_DEVICES=0,1 python test.py  # 使用多个 GPU
```

## 中断模式规则

### 当 `--no-interrupt` 生效时

遇到以下情况**不停下来询问用户**，而是：

| 情况 | 处理方式 |
|------|----------|
| 测试失败 | 记录失败原因，尝试自动修复，继续下一步 |
| 代码冲突 | 尝试合理解决，记录解决方案 |
| 不确定的实现细节 | 选择最合理的方案继续 |
| 执行错误 | 分析错误，尝试修复，记录问题 |

**自动决策原则**:
1. 优先保证功能正确性
2. 遵循现有代码风格
3. 选择简单直接的实现
4. 记录所有自动决策到 `progress.md`

### 当未指定 `--no-interrupt` 时

遇到以下情况**可以询问用户**：
- 多个实现方案需要选择
- 测试持续失败无法自动修复
- 发现 task_plan.md 中的问题或矛盾

## 执行记录

### 进度文件: progress.md

实时更新 `progress.md` 记录：

```markdown
## 执行进度

### Phase X: [名称]
- 状态: [进行中/完成/失败]
- 开始时间: [时间]
- 完成时间: [时间]
- 修改文件: [文件列表]
- 自动决策: [如果有]
- 问题记录: [如果有]
```

### 发现记录: findings.md

记录执行过程中的重要发现到 `findings.md`。

## 示例用法

```bash
# 使用 GPU 2，允许中断
/exec-plan --gpu 2

# 使用 GPU 0，不中断执行
/exec-plan --gpu 0 --no-interrupt

# 简短形式
/exec-plan -g 1 -n
```

## 完成标准

执行完成后，确保：

1. **所有 Phase 完成**: task_plan.md 中的所有 Phase 都已实施
2. **测试通过**: task_plan.md 中的测试计划全部通过
3. **代码质量**: 修改符合项目代码规范
4. **文档更新**: progress.md 包含完整执行记录

## 重要约束

1. **GPU 隔离**: 绝对不能使用指定 GPU 以外的设备
2. **遵循计划**: 严格按照 task_plan.md 执行，不做计划外的修改
3. **渐进式修改**: 每个 Phase 完成后验证，而不是最后一起验证
4. **回滚准备**: 重大修改前考虑是否需要 git commit 保存点



### File: .feynman/commands/ultra-think.md
---
description: Deep analysis and problem solving with multi-dimensional thinking
argument-hint: [problem or question to analyze]
---

# Deep Analysis and Problem Solving Mode

Deep analysis and problem solving mode

## Instructions

1. **Initialize Ultra Think Mode**
   - Acknowledge the request for enhanced analytical thinking
   - Set context for deep, systematic reasoning
   - Prepare to explore the problem space comprehensively

2. **Parse the Problem or Question**
   - Extract the core challenge from: $ARGUMENTS
   - Identify all stakeholders and constraints
   - Recognize implicit requirements and hidden complexities
   - Question assumptions and surface unknowns

3. **Multi-Dimensional Analysis**
   Approach the problem from multiple angles:
   
   ### Technical Perspective
   - Analyze technical feasibility and constraints
   - Consider scalability, performance, and maintainability
   - Evaluate security implications
   - Assess technical debt and future-proofing
   
   ### Business Perspective
   - Understand business value and ROI
   - Consider time-to-market pressures
   - Evaluate competitive advantages
   - Assess risk vs. reward trade-offs
   
   ### User Perspective
   - Analyze user needs and pain points
   - Consider usability and accessibility
   - Evaluate user experience implications
   - Think about edge cases and user journeys
   
   ### System Perspective
   - Consider system-wide impacts
   - Analyze integration points
   - Evaluate dependencies and coupling
   - Think about emergent behaviors

4. **Generate Multiple Solutions**
   - Brainstorm at least 3-5 different approaches
   - For each approach, consider:
     - Pros and cons
     - Implementation complexity
     - Resource requirements
     - Potential risks
     - Long-term implications
   - Include both conventional and creative solutions
   - Consider hybrid approaches

5. **Deep Dive Analysis**
   For the most promising solutions:
   - Create detailed implementation plans
   - Identify potential pitfalls and mitigation strategies
   - Consider phased approaches and MVPs
   - Analyze second and third-order effects
   - Think through failure modes and recovery

6. **Cross-Domain Thinking**
   - Draw parallels from other industries or domains
   - Apply design patterns from different contexts
   - Consider biological or natural system analogies
   - Look for innovative combinations of existing solutions

7. **Challenge and Refine**
   - Play devil's advocate with each solution
   - Identify weaknesses and blind spots
   - Consider "what if" scenarios
   - Stress-test assumptions
   - Look for unintended consequences

8. **Synthesize Insights**
   - Combine insights from all perspectives
   - Identify key decision factors
   - Highlight critical trade-offs
   - Summarize innovative discoveries
   - Present a nuanced view of the problem space

9. **Provide Structured Recommendations**
   Present findings in a clear structure:
   ```
   ## Problem Analysis
   - Core challenge
   - Key constraints
   - Critical success factors
   
   ## Solution Options
   ### Option 1: [Name]
   - Description
   - Pros/Cons
   - Implementation approach
   - Risk assessment
   
   ### Option 2: [Name]
   [Similar structure]
   
   ## Recommendation
   - Recommended approach
   - Rationale
   - Implementation roadmap
   - Success metrics
   - Risk mitigation plan
   
   ## Alternative Perspectives
   - Contrarian view
   - Future considerations
   - Areas for further research
   ```

10. **Meta-Analysis**
    - Reflect on the thinking process itself
    - Identify areas of uncertainty
    - Acknowledge biases or limitations
    - Suggest additional expertise needed
    - Provide confidence levels for recommendations

## Usage Examples

```bash
# Architectural decision
/ultra-think Should we migrate to microservices or improve our monolith?

# Complex problem solving
/ultra-think How do we scale our system to handle 10x traffic while reducing costs?

# Strategic planning
/ultra-think What technology stack should we choose for our next-gen platform?

# Design challenge
/ultra-think How can we improve our API to be more developer-friendly while maintaining backward compatibility?
```

## Key Principles

- **First Principles Thinking**: Break down to fundamental truths
- **Systems Thinking**: Consider interconnections and feedback loops
- **Probabilistic Thinking**: Work with uncertainties and ranges
- **Inversion**: Consider what to avoid, not just what to do
- **Second-Order Thinking**: Consider consequences of consequences

## Output Expectations

- Comprehensive analysis (typically 2-4 pages of insights)
- Multiple viable solutions with trade-offs
- Clear reasoning chains
- Acknowledgment of uncertainties
- Actionable recommendations
- Novel insights or perspectives


### File: .feynman/rules/doc-sync.md

# Documentation Sync Rule

## 强制规则 (Mandatory Rule)

**每当 `docs/` 目录下的文档发生任何结构性变更（新增、重命名、删除或核心意图改变）时，必须同步更新所有的环境配置入口文件。**

### 必须同步更新的文件清单

1. `CLAUDE.md` (为 Claude 实例提供索引)
2. `GEMINI.md` (为 Gemini 实例提供索引)
3. `AGENTS.md` (为 Feynman / 项目主控提供索引)

### 更新内容

必须在上述三个文件的 **Documentation Index** (或等效的表格) 中，精确添加或修改对应的条目：
- `Document` 列: 文档的相对路径，如 ``[`docs/new_doc.md`](docs/new_doc.md)``
- `Purpose` 列: 一句话概括文档的核心作用、涉及的系统机制或总结的数据指标。

### 触发条件

- 创建了新的解释文档、Benchmark 报告、Debug 记录或架构说明。
- 重构了现有的 Markdown 文档名称，或合并了多个文档。
- 废弃了旧的临时分析记录。

### 执行检查表 (Checklist)

- [ ] `docs/your_new_file.md` 已写入并完成最后审查。
- [ ] 已打开 `CLAUDE.md` 并更新 Index 表格。
- [ ] 已打开 `GEMINI.md` 并更新 Index 表格。
- [ ] 已打开 `AGENTS.md` 并更新 Index 表格。


