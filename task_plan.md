# Task Plan: Enable CUDA Graphs for CPU Offload Mode

## Problem Summary

Running `bench_offload.py` fails with:
```
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

**Root cause**: In offload mode, `HybridKVCacheManager.get_layer_cache()` returns empty tensors (by design), but CUDA graph capture calls `Attention.forward()` decode path which expects valid k_cache/v_cache.

**User requirement**: Enable CUDA graphs in offload mode for better decode performance.

## Deep Analysis: Why Current Design is Incompatible

### Current Offload Decode Flow (`run_layerwise_offload_decode`)

```
1. Preload N layers to ring buffer (H2D async)
2. For each layer:
   a. Wait for buffer load
   b. LayerNorm → QKV proj → RoPE
   c. k_full = torch.cat([k_prefill, k_decode_prev, k_new])  <-- DYNAMIC SHAPE
   d. flash_attn_varlen_func(q, k_full, v_full, ...)        <-- VARIABLE LENGTH
   e. O_proj → MLP
   f. Start next layer H2D load
3. Final norm → Logits → Sample
```

### CUDA Graph Incompatibility Points

| Issue | Location | Why Incompatible |
|-------|----------|------------------|
| Dynamic tensor creation | `torch.cat([k_prefill, ...])` | Creates new tensors with variable shapes |
| Variable-length attention | `flash_attn_varlen_func` | `max_seqlen_k` changes every step |
| Data-dependent branching | `if num_decode_tokens > 1` | Control flow varies at runtime |
| Empty k_cache/v_cache | `Attention.forward()` | Current capture uses standard decode path |

### Why Empty Tensors in Offload Mode?

`HybridKVCacheManager.get_layer_cache()` returns empty tensors because:
- Offload mode manages KV via `OffloadEngine`'s ring buffer
- The standard `Attention.forward()` is NEVER used in offload inference
- Empty tensors are intentional placeholders

## Solution: Fixed-Address CUDA Graph Capture for Offload Decode

### Key Insight

The `OffloadEngine` ring buffer already has **fixed GPU addresses** with **fixed max shape**:
```python
layer_k_cache: [num_kv_buffers, max_seq_len, kv_heads, head_dim]  # Fixed!
layer_v_cache: [num_kv_buffers, max_seq_len, kv_heads, head_dim]  # Fixed!
```

`flash_attn_with_kvcache` supports **cache_seqlens** parameter for variable actual lengths with fixed-shape cache. This is the key to CUDA graph compatibility!

### Solution Design

Replace `torch.cat` + `flash_attn_varlen_func` with:
1. Pre-copy decode buffer content to ring buffer at correct offset
2. Store new token KV directly to ring buffer
3. Use `flash_attn_with_kvcache` with `cache_seqlens` for variable length

```python
# Before (dynamic, not graphable):
k_full = torch.cat([k_prefill, k_decode_prev, k_new], dim=0)
o = flash_attn_varlen_func(q, k_full, v_full, ...)

# After (fixed addresses, graphable):
# Ring buffer already has k_prefill at [0:prefill_len]
# Copy decode_prev and k_new to buffer at [prefill_len:]
ring_buffer[prefill_len:prefill_len+decode_len] = decode_buffer
ring_buffer[total_len-1] = k_new

o = flash_attn_with_kvcache(
    q.unsqueeze(1),                    # [1, 1, heads, dim] - fixed shape
    ring_k.unsqueeze(0),               # [1, max_seq_len, heads, dim] - FIXED ADDRESS
    ring_v.unsqueeze(0),               # [1, max_seq_len, heads, dim] - FIXED ADDRESS
    cache_seqlens=total_tokens_tensor, # [1] - variable VALUE, fixed address
    softmax_scale=scale,
    causal=True,
)
```

## Implementation Plan

### Phase 1: Modify Offload Decode for CUDA Graph Compatibility

**File**: `nanovllm/engine/model_runner.py`

**Changes**:
1. Add `capture_offload_cudagraph()` method
2. Modify `run_layerwise_offload_decode()` to use fixed-address buffers
3. Replace `flash_attn_varlen_func` with `flash_attn_with_kvcache`

#### 1.1 New Method: `capture_offload_cudagraph()`

```python
@torch.inference_mode()
def capture_offload_cudagraph(self):
    """
    Capture CUDA graphs for offload decode.

    Key design:
    - Uses OffloadEngine's ring buffer as fixed-address k_cache/v_cache
    - Captures per-layer compute (after H2D load is done)
    - Uses flash_attn_with_kvcache with cache_seqlens for variable context
    """
    offload_engine = self.kvcache_manager.offload_engine
    num_layers = len(self.model.model.layers)
    num_buffers = offload_engine.num_kv_buffers
    max_seq_len = offload_engine.max_seq_len

    # Fixed-address tensors for graph capture
    input_ids = torch.zeros(1, dtype=torch.int64, device="cuda")
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    cache_seqlens = torch.zeros(1, dtype=torch.int32, device="cuda")
    hidden_output = torch.zeros(1, self.config.hf_config.hidden_size, device="cuda")

    # Graph capture per buffer slot (deterministic: layer_id % num_buffers)
    self.offload_graphs = {}
    self.offload_graph_pool = None

    for buffer_idx in range(num_buffers):
        graph = torch.cuda.CUDAGraph()

        # Get fixed-address ring buffer for this slot
        k_cache = offload_engine.layer_k_cache[buffer_idx:buffer_idx+1]  # [1, max_seq, heads, dim]
        v_cache = offload_engine.layer_v_cache[buffer_idx:buffer_idx+1]

        # Warmup
        with torch.cuda.stream(offload_engine.compute_stream):
            # ... (layer forward pass using k_cache, v_cache)
            pass

        # Capture
        with torch.cuda.graph(graph, self.offload_graph_pool):
            # ... (same layer forward pass)
            pass

        if self.offload_graph_pool is None:
            self.offload_graph_pool = graph.pool()
        self.offload_graphs[buffer_idx] = graph

    self.offload_graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        cache_seqlens=cache_seqlens,
        hidden_output=hidden_output,
    )
```

#### 1.2 Modified `run_layerwise_offload_decode()`

Key changes:
1. Copy decode buffer content to ring buffer before attention
2. Store new token directly to ring buffer
3. Use `flash_attn_with_kvcache` instead of `flash_attn_varlen_func`
4. Optionally use captured CUDA graph for per-layer compute

```python
# In the layer loop, replace:
k_full = torch.cat([k_prefill, k_decode_prev, k_new], dim=0)
attn_output = flash_attn_varlen_func(q, k_full, v_full, ...)

# With:
# 1. Get ring buffer slice
k_buffer = offload_engine.layer_k_cache[buffer_idx]  # [max_seq_len, heads, dim]
v_buffer = offload_engine.layer_v_cache[buffer_idx]

# 2. Copy decode buffer to ring buffer (after prefill content)
if num_decode_tokens > 1:
    k_buffer[total_prefill_tokens:total_prefill_tokens+num_decode_tokens-1].copy_(k_decode_prev)
    v_buffer[total_prefill_tokens:total_prefill_tokens+num_decode_tokens-1].copy_(v_decode_prev)

# 3. Store new token to ring buffer
total_kv_tokens = total_prefill_tokens + num_decode_tokens
k_buffer[total_kv_tokens-1].copy_(k_new.squeeze(0))
v_buffer[total_kv_tokens-1].copy_(v_new.squeeze(0))

# 4. Flash attention with fixed-address cache
cache_seqlens = torch.tensor([total_kv_tokens], dtype=torch.int32, device="cuda")
attn_output = flash_attn_with_kvcache(
    q.unsqueeze(1),           # [1, 1, heads, dim]
    k_buffer.unsqueeze(0),    # [1, max_seq_len, heads, dim] - FIXED ADDRESS
    v_buffer.unsqueeze(0),    # [1, max_seq_len, heads, dim] - FIXED ADDRESS
    cache_seqlens=cache_seqlens,
    softmax_scale=layer.self_attn.attn.scale,
    causal=True,
)
attn_output = attn_output.squeeze(1)  # [1, heads*dim]
```

### Phase 2: Handle CUDA Graph Capture in `__init__`

**File**: `nanovllm/engine/model_runner.py`

**Change**: Line 46-47

```python
# Current (crashes in offload mode):
if not self.enforce_eager:
    self.capture_cudagraph()

# Fixed (conditional capture based on mode):
if not self.enforce_eager:
    if config.enable_cpu_offload:
        self.capture_offload_cudagraph()  # New method for offload mode
    else:
        self.capture_cudagraph()  # Standard PagedAttention decode
```

### Phase 3: Per-Layer Graph vs Full-Decode Graph

Two approaches for graph capture:

#### Option A: Per-Layer Graphs (Simpler, Less Overhead Reduction)
- Capture N graphs (one per buffer slot)
- Each graph covers: LayerNorm → QKV → RoPE → Attention → O_proj → MLP
- H2D transfers and buffer management outside graph

#### Option B: Full-Decode Graph (More Complex, Maximum Overhead Reduction)
- Capture one graph for entire decode step (all layers)
- Requires all H2D loads completed before graph replay
- Better kernel fusion, less CPU overhead

**Recommendation**: Start with Option A (simpler), optimize to Option B later.

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Modify decode to use fixed-address buffers + flash_attn_with_kvcache | [ ] |
| Phase 2 | Add `capture_offload_cudagraph()` method | [ ] |
| Phase 3 | Update `__init__` to call correct capture method | [ ] |
| Phase 4 | Test with `bench_offload.py` | [ ] |
| Phase 5 | Benchmark performance improvement | [ ] |

## Key Code Changes Summary

| File | Change |
|------|--------|
| `model_runner.py:46-47` | Conditional CUDA graph capture based on offload mode |
| `model_runner.py` (new) | Add `capture_offload_cudagraph()` method |
| `model_runner.py:850-1010` | Modify `run_layerwise_offload_decode()` to use fixed-address attention |

## Alternative: Quick Fix (Skip Graph Capture)

If CUDA graph support is not immediately needed, the simpler fix is:

```python
# Line 46-47 in model_runner.py
if not self.enforce_eager and not config.enable_cpu_offload:
    self.capture_cudagraph()
```

This skips CUDA graph capture entirely in offload mode. Offload mode will use eager execution (which already works).

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| flash_attn_with_kvcache API differences | Test with actual flash-attn version |
| Memory overhead of fixed-size buffers | Already allocated in OffloadEngine |
| Performance regression | Benchmark before/after |
| Graph capture complexity | Start with per-layer graphs |

## Expected Performance Impact

| Metric | Without Graph | With Graph | Improvement |
|--------|---------------|------------|-------------|
| Decode latency per token | Baseline | ~10-20% faster | Reduced kernel launch overhead |
| GPU utilization | Medium | Higher | Better kernel fusion |
