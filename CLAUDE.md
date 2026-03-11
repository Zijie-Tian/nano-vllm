# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3, Llama-3, and GLM-4 models with CPU offload for long-context inference.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture_guide.md`](docs/architecture_guide.md) | Core components, CPU offload system design, ring buffer architecture, stream configuration |
| [`docs/sparse_policy_architecture.md`](docs/sparse_policy_architecture.md) | SparsePolicy abstraction: prefill/decode delegation, pipeline modes, policy implementations |
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

## Rules Index

| Rule | Purpose |
|------|---------|
| [`.claude/rules/multi-gpu-debugging.md`](.claude/rules/multi-gpu-debugging.md) | **Multi-GPU debugging**: GPU allocation (1-2 for validation, rest for exploration), single-task validation policy |
| [`.claude/rules/gpu-testing.md`](.claude/rules/gpu-testing.md) | GPU type detection, card assignment, needle test requirements |
| [`.claude/rules/sparse-policy.md`](.claude/rules/sparse-policy.md) | SparsePolicy implementation requirements |
| [`.claude/rules/planning-with-files.md`](.claude/rules/planning-with-files.md) | Planning file management for complex tasks |
| [`.claude/rules/gpu-monitor.md`](.claude/rules/gpu-monitor.md) | **GPU memory monitoring**: 必须使用 gpu-monitor agent，禁止手动 nvidia-smi 循环 |
| [`.claude/rules/test-ruler.md`](.claude/rules/test-ruler.md) | **test_ruler.py 规则**: 禁止 --help，必须查阅文档，含快速参考和命令模板 |
| **Documentation Sync** | Whenever a new document is added to `docs/`, its path and purpose **MUST** be immediately indexed in both `GEMINI.md` and `CLAUDE.md`. |

## GPU Mutex for Multi-Instance Debugging

**IMPORTANT**: When running multiple Claude instances for parallel debugging, different rules apply based on script type:

### Benchmarks (`bench*.py`) - Exclusive GPU Access Required

Before running any `bench*.py` script, Claude MUST wait for exclusive GPU access:

```bash
# Check and wait for GPU to be free
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "GPU busy, waiting 10s..."
  sleep 10
done
```

### Other Scripts (tests, examples) - No Special Requirements

For non-benchmark scripts, exclusive GPU access is NOT required. Multiple nanovllm processes can run simultaneously on different GPUs - each process automatically selects a unique port for `torch.distributed` communication.

## Multi-Instance Development with PYTHONPATH

**IMPORTANT**: When running multiple Claude instances on different worktrees, do NOT use `pip install -e .` globally as it will affect other instances.

**Use PYTHONPATH directly** - no pip install needed:

```bash
# Set PYTHONPATH to point to the project root directory
PYTHONPATH=/path/to/your/worktree:$PYTHONPATH python <script.py>

# Example: running tests
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_needle.py
```

**Benefits**:
- No `pip install` required
- Code changes take effect immediately (no reinstall needed)
- Each worktree is completely isolated

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 1024 | Tokens per block (4096 now works after race condition fix) |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |
| `enforce_eager` | False | Set True to disable CUDA graphs |

## Benchmarking

**Files**: `bench.py` (GPU), `bench_offload.py` (CPU offload), `bench_vllm.py` (comparison)

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

---

**Author**: Zijie Tian
docs/tmac_tvm_qgemm_migration_guide.md
