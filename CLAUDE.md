# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3 models with CPU offload for long-context inference.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture_guide.md`](docs/architecture_guide.md) | Core components, CPU offload system design, ring buffer architecture, stream configuration |
| [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md) | Block sparse attention methods (XAttention, FlexPrefill, MInference, AvgPool, Quest), computation flow, algorithms |
| [`docs/debugging_guide.md`](docs/debugging_guide.md) | PyTorch hooks for debugging, hook positions, tensor comparison, memory profiling |
| [`docs/optimization_guide.md`](docs/optimization_guide.md) | Performance optimizations: sgDMA (15x), Triton merge (4.3x), N-way pipeline (2x) |
| [`docs/known_issues.md`](docs/known_issues.md) | Documented bugs and fixes: partial last block bug, block size 4096 race condition |
| [`docs/ruler_benchmark_results_32k.md`](docs/ruler_benchmark_results_32k.md) | RULER benchmark results (32K context): 13 tasks, 92.3% accuracy, CPU offload performance |
| [`docs/ruler_32k_chunked_offload_issue.md`](docs/ruler_32k_chunked_offload_issue.md) | ⚠️ OPEN ISSUE: 32K chunked offload accuracy problem (35% error rate in RULER) |

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
