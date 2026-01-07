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

**Example workflow**:
```bash
# First check if GPU is in use
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader

# If output is empty, proceed with your command
python bench_offload.py

# If output shows processes, wait until they finish
```

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

**For shell session** (optional):
```bash
export PYTHONPATH=/path/to/your/worktree:$PYTHONPATH
python tests/test_needle.py  # PYTHONPATH already set
```

## Sparse Attention

For sparse attention related content (block sparse attention, MInference, FlexPrefill, XAttention, AvgPool, etc.), refer to [`docs/sparse_attention_guide.md`](docs/sparse_attention_guide.md).

### Quest Sparse Policy

**Files**: `nanovllm/kvcache/sparse/quest.py`, `nanovllm/kvcache/sparse/policy.py`

Quest policy selects Top-K blocks based on query-key similarity bounds using min/max key metadata.

**Scoring Mechanism**:
```python
score_min = torch.einsum('hd,bhd->bh', q, key_min)  # [num_blocks, kv_heads]
score_max = torch.einsum('hd,bhd->bh', q, key_max)  # [num_blocks, kv_heads]
scores = torch.maximum(score_min, score_max).mean(dim=-1)  # [num_blocks] ← averaged!
```

**Critical Limitation - No Per-Head Scheduling**:

The `.mean(dim=-1)` averages scores across all heads, making a **unified** block selection for all heads:

```
Block A: head0 needs (+4), head1 doesn't (-4) → avg = 0 → NOT selected
Block B: head0 doesn't (-4), head1 needs (+4) → avg = 0 → NOT selected
Block C: both heads moderately need (+2, +2) → avg = +2 → selected
```

**Why Per-Head Scheduling is Infeasible**:
1. **Memory Layout**: GPU cache stores all heads together `[block_size, kv_heads, head_dim]`
2. **FlashAttention**: Requires complete heads - partial heads cause dimension mismatch
3. **Block Granularity**: If any head needs a block, the entire block (all heads) must be loaded

**Policy Types**:
- `FullAttentionPolicy`: `supports_prefill=True, supports_decode=True` - loads all blocks
- `QuestPolicy`: `supports_prefill=False, supports_decode=True` - decode-only Top-K selection

## Architecture

### Core Components

- **LLMEngine** (`llm_engine.py`): Main entry, runs prefill-decode loop
- **ModelRunner** (`model_runner.py`): Loads weights, allocates KV cache, CUDA graphs, layer-wise offload
- **Scheduler** (`scheduler.py`): Two-phase scheduling (prefill → decode)
- **BlockManager** (`block_manager.py`): Paged attention with prefix caching (xxhash), default block size 4096
- **Attention** (`layers/attention.py`): FlashAttention for standard inference

## PyTorch Hooks for Debugging

### Hook Positions in Qwen3

```
decoder_layer
├── input_layernorm (RMSNorm)
├── self_attn (Qwen3Attention)          ← Hook here for attention I/O after o_proj
│   ├── q_proj → q_norm → RoPE
│   ├── k_proj → k_norm → RoPE
│   ├── v_proj
│   ├── attn (Attention)                ← Hook here for Q/K/V tensors
│   │   └── FlashAttention / SDPA
│   └── o_proj
├── post_attention_layernorm (RMSNorm)
└── mlp (Qwen3MLP)
```

### Hook Types & Data Shapes

| Hook Position | Type | Captured Data |
|---------------|------|---------------|
| `self_attn` | post | `[batch, seq_len, hidden_size]` - after o_proj |
| `self_attn.attn` | pre | Q,K,V: `[seq_len, num_heads, head_dim]` - after RoPE |
| `self_attn.attn` | post | `[seq_len, num_heads, head_dim]` - before o_proj |

### Example: Capture Attention Outputs

```python
storage = {}

def make_hook(layer_id: int, storage: dict):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        # nanovllm shape: [num_tokens, hidden_size] -> add batch dim
        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
        storage[layer_id] = attn_output.detach().clone()
    return hook

# Register hooks
hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(make_hook(layer_idx, storage)))

# Run inference...

# Cleanup
for hook in hooks:
    hook.remove()
```

### Reference Implementation

Key files:
- `tests/modeling_qwen3.py`: Reference Qwen3 implementation (torch + transformers only)
- `tests/test_needle_ref.py`: Reference needle test using custom Qwen3
- `tests/test_needle.py`: Needle-in-haystack test for nanovllm

### Common Pitfalls

1. **Shape mismatch**: nanovllm uses `[num_tokens, ...]` while torch uses `[batch, seq_len, ...]`
2. **Hook position**: `self_attn` captures after o_proj, `self_attn.attn` captures before o_proj
3. **Output format**: nanovllm returns tuple `(attn_output, None)`, handle with `output[0]`

## Layer-wise CPU Offload System

### Design Philosophy

Unlike chunked prefill (which processes chunks across all layers), **layer-wise offload** processes the entire sequence through one layer at a time:

```
Layer 0: [full sequence] → compute → offload K,V to CPU
Layer 1: [full sequence] → compute → offload K,V to CPU
...
Layer N: [full sequence] → compute → offload K,V to CPU
```

**Benefits**:
- Supports MInference sparse attention (requires full KV access per layer)
- Simpler memory management (one layer's KV in GPU at a time)
- Peak GPU memory = one layer's KV cache + attention workspace

### Key Files

- `nanovllm/engine/model_runner.py`: Main implementation (`run_layerwise_offload_prefill`, `run_layerwise_offload_decode`)
- `nanovllm/kvcache/hybrid_manager.py`: CPU block management helpers
- `nanovllm/kvcache/offload_engine.py`: CPU/GPU cache storage

### Memory Layout

**CPU Cache** (pinned memory):
```python
k_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
v_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
```

**Per-layer KV size** (Qwen3-4B: 8 kv_heads × 128 head_dim × 2 bytes × 2 for K+V = 4KB/token):

| Context Length | KV per Layer |
|----------------|--------------|
| 128K tokens | 512 MB |
| 256K tokens | 1 GB |
| 512K tokens | 2 GB |
| 1M tokens | 4 GB |

### Prefill Flow

```python
def run_layerwise_offload_prefill(self, seqs: list[Sequence]) -> list[int]:
    # 1. Embedding
    hidden_states = self.model.model.embed_tokens(input_ids)

    # 2. Process each layer
    for layer_id in range(num_layers):
        # QKV projection + norms + RoPE
        q = apply_rotary_pos_emb(q_proj(hidden_states), cos, sin)
        k = apply_rotary_pos_emb(k_proj(hidden_states), cos, sin)
        v = v_proj(hidden_states)

        # Full FlashAttention (entire sequence)
        attn_out = flash_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen, causal=True)

        # MLP
        hidden_states = mlp(attn_out + residual)

        # Synchronous offload to CPU (CRITICAL: must be sync to avoid memory reuse bugs)
        self._offload_layer_kv_to_cpu_sync(layer_id, k, v, cpu_block_ids, total_tokens)

    # 3. Final norm + sampling
    return sampled_tokens
```

### Decode Flow

```python
def run_layerwise_offload_decode(self, seqs: list[Sequence]) -> list[int]:
    # For each layer:
    for layer_id in range(num_layers):
        # 1. Load all prefilled KV from CPU
        for block_idx, cpu_block_id in enumerate(cpu_block_table):
            k_block = offload_engine.k_cache_cpu[layer_id, cpu_block_id, :valid_tokens].to("cuda")
            v_block = offload_engine.v_cache_cpu[layer_id, cpu_block_id, :valid_tokens].to("cuda")

        # 2. Compute new Q,K,V for current token
        q_new = apply_rotary_pos_emb(q_proj(hidden_states), cos, sin)
        k_new = apply_rotary_pos_emb(k_proj(hidden_states), cos, sin)
        v_new = v_proj(hidden_states)

        # 3. Concatenate and compute attention
        k_full = torch.cat([k_prefill, k_new], dim=0)
        v_full = torch.cat([v_prefill, v_new], dim=0)
        attn_out = flash_attn_varlen_func(q_new, k_full, v_full, ..., causal=False)
        # Note: causal=False because single query token should attend to ALL keys
```

### Critical Implementation Details

**1. Synchronous Offload Required**

Async offload with `non_blocking=True` causes memory reuse bugs:
```python
# BUG: PyTorch may reuse k,v GPU memory before async copy completes
offload_engine.k_cache_cpu[layer_id, block_id].copy_(k[start:end], non_blocking=True)

# CORRECT: Synchronous copy ensures data integrity
offload_engine.k_cache_cpu[layer_id, block_id, :size].copy_(k[start:end])  # sync
```

**2. Decode Attention: causal=False**

During decode, the single query token must attend to ALL keys (not just preceding ones):
```python
# Prefill: causal=True (each token only attends to previous tokens)
attn_out = flash_attn_varlen_func(..., causal=True)

# Decode: causal=False (query at position N attends to all N-1 prefill + itself)
attn_out = flash_attn_varlen_func(..., causal=False)
```

### Helper Methods in HybridKVCacheManager

```python
# Get all CPU blocks for a sequence
cpu_blocks = manager.get_all_cpu_blocks(seq)  # List[int]

# Get only prefilled (offloaded) CPU blocks
prefilled_blocks = manager.get_prefilled_cpu_blocks(seq)  # List[int]

# Get cached prefill length (doesn't change during decode)
prefill_len = manager.get_prefill_len(seq)  # int

# Get decode start position
decode_pos = manager.get_decode_start_pos(seq)  # int
```

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kvcache_block_size` | 4096 | Tokens per block |
| `max_num_batched_tokens` | 16384 | Set = max_model_len for long context |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction |
| `enable_cpu_offload` | False | Enable for long context |

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
