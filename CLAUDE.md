# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports multiple model architectures (Qwen3, Qwen2, Llama) with CPU offload for long-context inference.

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

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture_guide.md`](docs/architecture_guide.md) | Core components, layer-wise CPU offload design, prefill/decode flows, implementation details |
| [`docs/multi_model_support.md`](docs/multi_model_support.md) | Model registry system, adding new models (Qwen3/Llama), architecture differences, RoPE scaling |
| [`docs/cuda_graph_offload_guide.md`](docs/cuda_graph_offload_guide.md) | CUDA graph support for CPU offload decode path, 4x decode speedup |
| [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md) | Block sparse attention methods (MInference, FlexPrefill, XAttention, Quest), computation flow |
| [`docs/block_sparse_attention_lib.md`](docs/block_sparse_attention_lib.md) | MIT-Han-Lab Block-Sparse-Attention library reference: sparse modes, API, performance |
| [`docs/sparse_prefill_integration_plan.md`](docs/sparse_prefill_integration_plan.md) | Integration plan for MInference/XAttention/FlexPrefill with unified BlockMask interface |
| [`docs/sparse_offload_integration.md`](docs/sparse_offload_integration.md) | Sparse policy integration with layerwise offload, `requires_block_selection` interface design |
| [`docs/layerwise_offload_memory_analysis.md`](docs/layerwise_offload_memory_analysis.md) | Memory allocation analysis with theoretical formulas and empirical validation (< 5% error) |
| [`docs/debugging_guide.md`](docs/debugging_guide.md) | PyTorch hooks for debugging, tensor comparison, memory profiling |
| [`docs/gpu_only_performance_issue.md`](docs/gpu_only_performance_issue.md) | GPU-only mode slower than offload due to PagedAttention scatter overhead, optimization proposals |
| [`docs/offload_accuracy_issue.md`](docs/offload_accuracy_issue.md) | **BUG**: CPU offload mode 66% accuracy vs 100% non-offload on RULER NIAH benchmark |
| [`docs/mst_memory_overhead_issue.md`](docs/mst_memory_overhead_issue.md) | **BUG**: MST integration shows ~18GB unexplained memory overhead. **Includes**: Root cause analysis (layer-wise processing), Appendix A: CUDA Graph memory deep dive (~13GB breakdown and why it's needed) |
| [`docs/64k_memory_analysis.md`](docs/64k_memory_analysis.md) | 64k inference memory analysis: GPU-only vs offload, OOM root cause (fragmentation), RTX 3090 limitations |
| [`docs/xattention_integration.md`](docs/xattention_integration.md) | XAttention integration guide: algorithm, implementation, design decisions, and testing |
| [`docs/xattention_analysis.md`](docs/xattention_analysis.md) | XAttention algorithm analysis: chunked estimation, block sparse attention, integration design |
| [`docs/development_notes.md`](docs/development_notes.md) | Development notes and scratchpad for ongoing work |

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 4096 | Tokens per block |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |
| `num_gpu_blocks` | 2 | GPU blocks for offload mode |
| `num_kv_buffers` | 4 | Ring buffer size (1-4), lower = less memory but slower decode |
| `enforce_eager` | False | Set True to disable CUDA graphs |
| `enable_mst` | False | Mini-Sequence Transformer for MLP memory reduction (16x at 128k) |
| `mst_chunk_size` | 8192 | Chunk size for MST (default: 8192 for 16x reduction at 128k) |
| `mst_min_seq_len` | 32768 | Minimum sequence length to activate MST |

## Benchmarking

**Files**: `bench.py` (GPU), `bench_offload.py` (CPU offload), `bench_vllm.py` (comparison)

**Common Issues**:
1. `max_num_batched_tokens < max_model_len`: Set equal for long context
2. CUDA graph dimension mismatch: Ensure `input_len + output_len <= max_model_len`
3. RoPE out of bounds: Check model's `max_position_embeddings` in config.json

**Model Limits**:
- Qwen3-0.6B/4B: 40960 tokens
- Qwen2.5-7B-Instruct-1M: 1048576 tokens
- Llama-3.1-8B-Instruct: 131072 tokens
- **64k on RTX 3090/4090 (24GB)**: Requires CPU offload + optimizations, see [`docs/64k_memory_analysis.md`](docs/64k_memory_analysis.md)

**Performance (Qwen3-4B, CPU Offload)**:
- Prefill: ~5700-8000 tok/s (varies by context length)
- Decode with CUDA Graph: ~50 tok/s (TPOT ~19ms)
- Decode Eager Mode: ~12 tok/s (TPOT ~80ms)
- **CUDA Graph speedup: 4x decode throughput**

---

**Author**: Zijie Tian
