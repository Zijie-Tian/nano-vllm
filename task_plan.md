# Task Plan: Fix GPU-only Mode Performance Issue

## Goal
Eliminate the `store_kvcache` scatter overhead in GPU-only mode by using **contiguous KV cache layout** (like offload mode), avoiding PagedAttention's blocked layout for single-sequence inference.

## Problem Summary

| Mode | Prefill Speed (32K, Qwen3-4B) |
|------|-------------------------------|
| GPU-only + MInference | 3383 tok/s |
| Offload + MInference | 5373 tok/s |

**Root cause**: PagedAttention's blocked layout requires expensive `index_copy_` scatter operations.

## Key Insight: Why Offload is Fast

Offload mode uses **contiguous layout** for KV cache:

```python
# OffloadEngine's CPU cache layout
k_cache_cpu: [num_layers, num_blocks, block_size, kv_heads, head_dim]

# Store is simple contiguous slice assignment
self.k_cache_cpu[layer_id, block_id, :actual_size].copy_(k[start:end])
```

The K,V computed during prefill `[seq_len, kv_heads, head_dim]` matches the cache layout - no format conversion needed!

## Solution: Contiguous Layout for GPU-only Mode

For GPU-only single-sequence mode, use the **same contiguous layout as offload mode**, but on GPU:

```
Current GPU-only (PagedAttention):
  Cache: [num_blocks, block_size, kv_heads, head_dim] (blocked)
  Store: scatter via index_copy_ (SLOW)

Proposed GPU-only (Contiguous):
  Cache: [num_layers, max_seq_len, kv_heads, head_dim] (contiguous)
  Store: slice assignment k_cache[layer_id, :seq_len] = k (FAST)
```

## Phases
- [x] Phase 1: Understand the problem and offload architecture
- [x] Phase 2: Add contiguous GPU KV cache in GPUOnlyManager
- [x] Phase 3: Implement `run_gpu_only_prefill()` using contiguous cache
- [x] Phase 4: Implement `run_gpu_only_decode()` using contiguous cache
- [x] Phase 5: Test and validate performance

## Detailed Design

### Phase 2: Contiguous GPU KV Cache

**File**: `nanovllm/kvcache/gpu_manager.py`

Add contiguous cache allocation for single-sequence mode:

```python
class GPUOnlyManager(KVCacheManager):
    def __init__(self, num_blocks: int, block_size: int, max_seq_len: int = 0):
        # ... existing code ...
        self.max_seq_len = max_seq_len

        # Contiguous cache for single-seq mode (allocated in allocate_cache)
        self.contiguous_k_cache = None  # [num_layers, max_seq_len, kv_heads, head_dim]
        self.contiguous_v_cache = None
        self.contiguous_seq_len = 0  # Track current sequence length

    def allocate_cache(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Existing PagedAttention cache for multi-seq/decode (keep for compatibility)
        self.kv_cache = torch.empty(
            2, num_layers, self._num_blocks, self._block_size,
            num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # Contiguous cache for single-seq prefill (if max_seq_len specified)
        if self.max_seq_len > 0:
            self.contiguous_k_cache = torch.empty(
                num_layers, self.max_seq_len, num_kv_heads, head_dim,
                dtype=dtype, device="cuda"
            )
            self.contiguous_v_cache = torch.empty(
                num_layers, self.max_seq_len, num_kv_heads, head_dim,
                dtype=dtype, device="cuda"
            )
```

**File**: `nanovllm/kvcache/__init__.py`

Update factory to pass max_seq_len:

```python
def create_kvcache_manager(config: "Config") -> KVCacheManager:
    if not getattr(config, 'enable_cpu_offload', False):
        return GPUOnlyManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            max_seq_len=config.max_model_len,  # NEW: pass max_seq_len
        )
    # ... rest unchanged
```

### Phase 3: Layer-wise GPU-only Prefill

**File**: `nanovllm/engine/model_runner.py`

Following offload pattern exactly - store K,V per-layer to contiguous cache:

```python
@torch.inference_mode()
def run_gpu_only_prefill(self, seqs: list[Sequence]) -> list[int]:
    """
    GPU-only prefill with contiguous KV cache layout.
    Mirrors run_layerwise_offload_prefill() but stores to GPU instead of CPU.
    """
    assert len(seqs) == 1, "GPU-only layer-wise prefill only supports single sequence"
    seq = seqs[0]

    num_layers = len(self.model.model.layers)
    total_tokens = len(seq)

    # Get contiguous GPU cache
    k_cache = self.kvcache_manager.contiguous_k_cache
    v_cache = self.kvcache_manager.contiguous_v_cache

    # Prepare inputs
    input_ids = torch.tensor(seq[:], dtype=torch.int64, device="cuda")
    positions = torch.arange(total_tokens, dtype=torch.int64, device="cuda")

    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

    # Embedding
    hidden_states = self.model.model.embed_tokens(input_ids)
    residual = None

    # Layer-by-layer processing (same as offload prefill)
    for layer_id in range(num_layers):
        layer = self.model.model.layers[layer_id]

        # Input LayerNorm
        if residual is None:
            hidden_ln, residual = layer.input_layernorm(hidden_states), hidden_states
        else:
            hidden_ln, residual = layer.input_layernorm(hidden_states, residual)

        # QKV projection
        qkv = layer.self_attn.qkv_proj(hidden_ln)
        q, k, v = qkv.split([
            layer.self_attn.q_size,
            layer.self_attn.kv_size,
            layer.self_attn.kv_size
        ], dim=-1)

        q = q.view(total_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
        k = k.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
        v = v.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

        # Q/K norms (Qwen3 specific)
        if not layer.self_attn.qkv_bias:
            q = layer.self_attn.q_norm(q.reshape(-1, layer.self_attn.head_dim))
            q = q.view(total_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k = layer.self_attn.k_norm(k.reshape(-1, layer.self_attn.head_dim))
            k = k.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

        # RoPE
        q, k = layer.self_attn.rotary_emb(positions, q, k)

        # Store K,V to contiguous GPU cache (same layout - no conversion!)
        k_cache[layer_id, :total_tokens] = k
        v_cache[layer_id, :total_tokens] = v

        # Sparse or Full attention (uses k, v directly)
        if self.sparse_prefill_policy is not None:
            attn_output = self.sparse_prefill_policy.sparse_prefill_attention(
                q, k, v, layer_id
            )
        else:
            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=total_tokens,
                max_seqlen_k=total_tokens,
                softmax_scale=layer.self_attn.attn.scale,
                causal=True,
            )

        # O projection
        attn_output = attn_output.view(total_tokens, -1)
        hidden_states = layer.self_attn.o_proj(attn_output)

        # Post-attention LayerNorm + MLP
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    # Final norm
    hidden_states, _ = self.model.model.norm(hidden_states, residual)

    # Compute logits
    logits = self.model.compute_logits(hidden_states[-1:])

    # Record prefill length for decode
    self.kvcache_manager.contiguous_seq_len = total_tokens

    # Sample
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    return token_ids
```

### Phase 4: Decode with Contiguous Cache

```python
@torch.inference_mode()
def run_gpu_only_decode(self, seqs: list[Sequence]) -> list[int]:
    """
    Decode using contiguous GPU KV cache.
    Similar to offload decode but simpler - all KV already on GPU.
    """
    assert len(seqs) == 1
    seq = seqs[0]

    num_layers = len(self.model.model.layers)
    k_cache = self.kvcache_manager.contiguous_k_cache
    v_cache = self.kvcache_manager.contiguous_v_cache
    context_len = self.kvcache_manager.contiguous_seq_len

    # Prepare inputs
    input_ids = torch.tensor([seq.last_token], dtype=torch.int64, device="cuda")
    positions = torch.tensor([len(seq) - 1], dtype=torch.int64, device="cuda")

    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    # Embedding
    hidden_states = self.model.model.embed_tokens(input_ids)
    residual = None

    for layer_id in range(num_layers):
        layer = self.model.model.layers[layer_id]

        # Input LayerNorm
        if residual is None:
            hidden_ln, residual = layer.input_layernorm(hidden_states), hidden_states
        else:
            hidden_ln, residual = layer.input_layernorm(hidden_states, residual)

        # QKV projection
        qkv = layer.self_attn.qkv_proj(hidden_ln)
        q, k_new, v_new = qkv.split([
            layer.self_attn.q_size,
            layer.self_attn.kv_size,
            layer.self_attn.kv_size
        ], dim=-1)

        q = q.view(1, layer.self_attn.num_heads, layer.self_attn.head_dim)
        k_new = k_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
        v_new = v_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

        # Q/K norms
        if not layer.self_attn.qkv_bias:
            q = layer.self_attn.q_norm(q.reshape(-1, layer.self_attn.head_dim))
            q = q.view(1, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k_new = layer.self_attn.k_norm(k_new.reshape(-1, layer.self_attn.head_dim))
            k_new = k_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

        # RoPE
        q, k_new = layer.self_attn.rotary_emb(positions, q, k_new)

        # Store new K,V to cache
        k_cache[layer_id, context_len] = k_new.squeeze(0)
        v_cache[layer_id, context_len] = v_new.squeeze(0)

        # Full K,V for attention (direct slice - no scatter!)
        k_full = k_cache[layer_id, :context_len + 1]
        v_full = v_cache[layer_id, :context_len + 1]

        # Attention
        cu_seqlens_k = torch.tensor([0, context_len + 1], dtype=torch.int32, device="cuda")
        attn_output = flash_attn_varlen_func(
            q, k_full, v_full,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=context_len + 1,
            softmax_scale=layer.self_attn.attn.scale,
            causal=False,
        )

        # O projection
        attn_output = attn_output.view(1, -1)
        hidden_states = layer.self_attn.o_proj(attn_output)

        # Post-attention LayerNorm + MLP
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    # Update context length
    self.kvcache_manager.contiguous_seq_len = context_len + 1

    # Final norm
    hidden_states, _ = self.model.model.norm(hidden_states, residual)

    # Compute logits
    logits = self.model.compute_logits(hidden_states)

    # Sample
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    return token_ids
```

### Phase 5: Decision Logic

```python
def _should_use_contiguous_gpu_mode(self, seqs: list[Sequence], is_prefill: bool) -> bool:
    """Check if contiguous GPU mode should be used."""
    # Must have contiguous cache allocated
    if not hasattr(self.kvcache_manager, 'contiguous_k_cache'):
        return False
    if self.kvcache_manager.contiguous_k_cache is None:
        return False

    # Must NOT be offload mode
    if hasattr(self.kvcache_manager, 'offload_engine'):
        return False

    # Single sequence only
    if len(seqs) != 1:
        return False

    # For prefill: has blocks (not warmup)
    if is_prefill and not seqs[0].block_table:
        return False

    return True


def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    # Check offload mode (existing)
    if hasattr(self, 'kvcache_manager') and hasattr(self.kvcache_manager, 'offload_engine'):
        use_layerwise_offload = self._should_use_layerwise_offload(seqs, is_prefill)
        if use_layerwise_offload:
            if is_prefill:
                return self.run_layerwise_offload_prefill(seqs)
            else:
                return self.run_layerwise_offload_decode(seqs)

    # Check contiguous GPU mode (NEW)
    if self._should_use_contiguous_gpu_mode(seqs, is_prefill):
        if is_prefill:
            return self.run_gpu_only_prefill(seqs)
        else:
            return self.run_gpu_only_decode(seqs)

    # Standard PagedAttention path
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    ...
```

## Architecture Comparison

| Aspect | Offload Mode | GPU-only (Proposed) | GPU-only (Current) |
|--------|--------------|---------------------|-------------------|
| Cache location | CPU (contiguous) | GPU (contiguous) | GPU (PagedAttention) |
| Cache layout | `[layers, blocks, block_size, heads, dim]` | `[layers, max_seq_len, heads, dim]` | `[blocks, block_size, heads, dim]` |
| Prefill store | Contiguous slice copy | **Slice assignment** | Scatter (index_copy_) |
| Decode read | H2D ring buffer | Direct GPU slice | PagedAttention |

## Memory Usage

Contiguous GPU cache: `2 * num_layers * max_seq_len * kv_heads * head_dim * dtype_size`

For Qwen3-4B with 32K max_seq_len:
- `2 * 28 * 32768 * 8 * 128 * 2 = 3.5GB`

Same as offload mode's CPU cache, but on GPU.

## Files to Modify

| File | Changes |
|------|---------|
| `nanovllm/kvcache/gpu_manager.py` | Add `max_seq_len`, contiguous cache allocation |
| `nanovllm/kvcache/__init__.py` | Pass `max_model_len` to GPUOnlyManager |
| `nanovllm/engine/model_runner.py` | Add `run_gpu_only_prefill()`, `run_gpu_only_decode()`, `_should_use_contiguous_gpu_mode()`, modify `run()` |

## Actual Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU-only prefill (32K) | 3383 tok/s | 3611 tok/s | ~7% |
| Decode | Baseline | Similar | ~0% |

Note: Expected ~60%+ improvement not achieved. The contiguous cache optimization provides modest improvement. The offload mode (5468 tok/s) benefits from async D2H transfer overlap which doesn't apply to GPU-only mode.

## Status
**All phases completed** - GPU-only contiguous cache implemented and test passes
