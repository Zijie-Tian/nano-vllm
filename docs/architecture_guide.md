# Architecture Guide

This document describes the core architecture and layer-wise CPU offload system of nano-vLLM.

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **LLMEngine** | `llm_engine.py` | Main entry, runs prefill-decode loop |
| **ModelRunner** | `model_runner.py` | Loads weights, allocates KV cache, CUDA graphs, layer-wise offload |
| **Scheduler** | `scheduler.py` | Two-phase scheduling (prefill → decode) |
| **BlockManager** | `block_manager.py` | Paged attention with prefix caching (xxhash), default block size 4096 |
| **Attention** | `layers/attention.py` | FlashAttention for standard inference |

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

| File | Purpose |
|------|---------|
| `nanovllm/engine/model_runner.py` | Main implementation (`run_layerwise_offload_prefill`, `run_layerwise_offload_decode`) |
| `nanovllm/kvcache/hybrid_manager.py` | CPU block management helpers |
| `nanovllm/kvcache/offload_engine.py` | CPU/GPU cache storage, ring buffer, async transfers |

### Memory Layout

**CPU Cache** (pinned memory):
```python
k_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
v_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
```

**GPU Ring Buffer** (for decode H2D pipeline):
```python
layer_k_cache: [num_kv_buffers, max_seq_len, kv_heads, head_dim]
layer_v_cache: [num_kv_buffers, max_seq_len, kv_heads, head_dim]
```

**Per-layer KV size** (Qwen3-4B: 8 kv_heads × 128 head_dim × 2 bytes × 2 for K+V = 4KB/token):

| Context Length | KV per Layer |
|----------------|--------------|
| 128K tokens | 512 MB |
| 256K tokens | 1 GB |
| 512K tokens | 2 GB |
| 1M tokens | 4 GB |

---

## Prefill Flow

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

---

## Decode Flow

```python
def run_layerwise_offload_decode(self, seqs: list[Sequence]) -> list[int]:
    # Ring buffer pipeline: preload first N layers
    for i in range(num_buffers):
        offload_engine.load_layer_kv_to_buffer(i, i, cpu_block_table, valid_tokens)

    # For each layer:
    for layer_id in range(num_layers):
        current_buffer = layer_id % num_buffers

        # 1. Wait for buffer load to complete
        offload_engine.wait_buffer_load(current_buffer)

        # 2. Get prefilled KV from ring buffer
        k_prefill, v_prefill = offload_engine.get_buffer_kv(current_buffer, total_prefill_tokens)

        # 3. Compute new Q,K,V for current token
        q_new = apply_rotary_pos_emb(q_proj(hidden_states), cos, sin)
        k_new = apply_rotary_pos_emb(k_proj(hidden_states), cos, sin)
        v_new = v_proj(hidden_states)

        # 4. Concatenate and compute attention
        k_full = torch.cat([k_prefill, k_new], dim=0)
        v_full = torch.cat([v_prefill, v_new], dim=0)
        attn_out = flash_attn_varlen_func(q_new, k_full, v_full, ..., causal=False)
        # Note: causal=False because single query token should attend to ALL keys

        # 5. Mark buffer done, start loading next layer
        offload_engine.record_buffer_compute_done(current_buffer)
        if layer_id + num_buffers < num_layers:
            offload_engine.load_layer_kv_to_buffer(current_buffer, layer_id + num_buffers, ...)
```

---

## Critical Implementation Details

### 1. Synchronous Offload Required

Async offload with `non_blocking=True` causes memory reuse bugs:

```python
# BUG: PyTorch may reuse k,v GPU memory before async copy completes
offload_engine.k_cache_cpu[layer_id, block_id].copy_(k[start:end], non_blocking=True)

# CORRECT: Synchronous copy ensures data integrity
offload_engine.k_cache_cpu[layer_id, block_id, :size].copy_(k[start:end])  # sync
```

### 2. Decode Attention: causal=False

During decode, the single query token must attend to ALL keys (not just preceding ones):

```python
# Prefill: causal=True (each token only attends to previous tokens)
attn_out = flash_attn_varlen_func(..., causal=True)

# Decode: causal=False (query at position N attends to all N-1 prefill + itself)
attn_out = flash_attn_varlen_func(..., causal=False)
```

### 3. Ring Buffer Synchronization

The ring buffer pipeline requires careful ordering:

```python
# CORRECT order:
offload_engine.store_decode_kv(layer_id, pos, k_new, v_new)  # Store new KV
offload_engine.record_buffer_compute_done(current_buffer)     # Mark done FIRST
offload_engine.load_layer_kv_to_buffer(...)                   # THEN start next load

# BUG: Starting load before marking done causes race condition
offload_engine.load_layer_kv_to_buffer(...)  # WRONG: buffer still in use!
offload_engine.record_buffer_compute_done(current_buffer)
```

---

## Helper Methods in HybridKVCacheManager

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
