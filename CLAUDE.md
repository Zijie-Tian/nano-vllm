# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines) for fast offline LLM inference. Supports Qwen3 models with CPU offload for long-context inference.

## GPU Mutex for Multi-Instance Debugging

**IMPORTANT**: When running multiple Claude instances for parallel debugging, only one GPU (cuda:0) is available. Before executing ANY command that uses the GPU (python scripts, benchmarks, tests), Claude MUST:

1. **Check GPU availability** by running:
   ```bash
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
   ```

2. **If processes are running on GPU**:
   - Wait and retry every 10 seconds until GPU is free
   - Use this polling loop:
     ```bash
     while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
       echo "GPU busy, waiting 10s..."
       sleep 10
     done
     ```

3. **Only proceed** when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` returns empty output

**Note**: This applies to ALL GPU operations including:
- Running tests (`python tests/test_*.py`)
- Running benchmarks (`python bench*.py`)
- Running examples (`python example.py`)
- Any script that imports torch/cuda

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
| [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md) | Block sparse attention methods (MInference, FlexPrefill, XAttention, Quest), computation flow |
| [`docs/layerwise_offload_memory_analysis.md`](docs/layerwise_offload_memory_analysis.md) | Memory allocation analysis with theoretical formulas and empirical validation (< 5% error) |
| [`docs/debugging_guide.md`](docs/debugging_guide.md) | PyTorch hooks for debugging, tensor comparison, memory profiling |

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 4096 | Tokens per block |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |
| `num_gpu_blocks` | 2 | GPU blocks for offload mode |
| `num_kv_buffers` | 4 | Ring buffer size for decode pipeline |

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
