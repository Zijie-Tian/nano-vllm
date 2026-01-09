# Task Plan: Enable CUDA Graphs for CPU Offload Mode

## Current Status: ✅ COMPLETED

### Phase 0 Completed: Refactor Offload Decode to Use Standard Attention Path

### Phases 1-3 Completed: CUDA Graph Support for Offload Mode

**Implementation**: Added per-layer CUDA graph capture and replay for offload decode path.

**Key Changes**:
1. `capture_offload_cudagraph()` captures one graph per transformer layer
2. Each graph uses the corresponding ring buffer slot based on `layer_id % num_buffers`
3. `run_layerwise_offload_decode()` replays graphs when `enforce_eager=False`
4. Synchronization added between graph replays to ensure correct data flow

**Test Results**:
- `test_needle.py --input-len 32768 --enable-offload --use-cuda-graph`: **PASSED**

---

### Previous Work: Refactor Offload Decode to Use Standard Attention Path

**Problem solved**: The original offload decode (`run_layerwise_offload_decode`) bypassed `Attention.forward()` by manually calling attention components. This was inconsistent with the standard execution path.

**Solution implemented**: Refactored to use `layer.forward()` which goes through:
```
Qwen3DecoderLayer.forward()
  → Qwen3Attention.forward()
    → Attention.forward()  ← Now properly used!
```

### Code Changes Made

**File**: `nanovllm/engine/model_runner.py`

1. **`run_layerwise_offload_decode()` (line 841-991)** - Completely refactored:

   Before (bypassed Attention):
   ```python
   qkv = layer.self_attn.qkv_proj(hidden_ln)
   q, k_new, v_new = qkv.split(...)
   q = layer.self_attn.q_norm(...)
   k = layer.self_attn.k_norm(...)
   q, k = layer.self_attn.rotary_emb(...)
   attn_output = flash_attn_varlen_func(q, k_full, v_full, ...)  # Direct call!
   hidden_states = layer.self_attn.o_proj(attn_output)
   ```

   After (uses standard path):
   ```python
   # Set up Attention module's cache to ring buffer
   attn_module.k_cache = offload_engine.layer_k_cache[buffer_idx:buffer_idx+1]
   attn_module.v_cache = offload_engine.layer_v_cache[buffer_idx:buffer_idx+1]

   # Set context for contiguous mode
   set_context(is_prefill=False, slot_mapping=..., context_lens=..., block_tables=None)

   # Standard layer forward - goes through Attention.forward()!
   hidden_states, residual = layer(positions, hidden_states, residual)
   ```

2. **`ModelRunner.__init__()` (line 46-57)** - Conditional CUDA graph capture:
   ```python
   if not self.enforce_eager:
       if config.enable_cpu_offload:
           # TODO: Implement capture_offload_cudagraph()
           pass  # Temporarily use eager execution
       else:
           self.capture_cudagraph()
   ```

### Test Results

| Test | Mode | Status |
|------|------|--------|
| `test_needle.py --input-len 4096` | GPU-only | PASSED |
| `test_needle.py --input-len 4096 --enable-offload` | CPU offload | PASSED |

## Remaining Work: Implement Offload CUDA Graph

### Why Standard `capture_cudagraph()` Cannot Be Used

The standard capture function captures the PagedAttention decode path:
```python
# capture_cudagraph() sets up:
k_cache: [num_blocks, block_size, kv_heads, head_dim]  # PagedAttention format
block_tables: [...] # Block indices for paged indexing
```

But offload mode uses contiguous ring buffer:
```python
# Offload decode sets up:
k_cache: [1, max_seq_len, kv_heads, head_dim]  # Contiguous format
block_tables: None  # No paging
```

### Implementation Plan for `capture_offload_cudagraph()`

#### Phase 1: Prepare Fixed-Address Tensors

```python
@torch.inference_mode()
def capture_offload_cudagraph(self):
    """Capture CUDA graphs for offload decode using ring buffer."""
    offload_engine = self.kvcache_manager.offload_engine
    num_buffers = offload_engine.num_kv_buffers

    # Fixed-address tensors for graph capture
    input_ids = torch.zeros(1, dtype=torch.int64, device="cuda")
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    slot_mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
    context_lens = torch.zeros(1, dtype=torch.int32, device="cuda")

    self.offload_graphs = {}
    self.offload_graph_pool = None
```

#### Phase 2: Capture Per-Buffer Graphs

Since layer processing rotates through ring buffers (`layer_id % num_buffers`), we need graphs for each buffer slot:

```python
    for buffer_idx in range(num_buffers):
        graph = torch.cuda.CUDAGraph()

        # Set Attention cache to this buffer slot (fixed address)
        for layer in self.model.model.layers:
            layer.self_attn.attn.k_cache = offload_engine.layer_k_cache[buffer_idx:buffer_idx+1]
            layer.self_attn.attn.v_cache = offload_engine.layer_v_cache[buffer_idx:buffer_idx+1]

        # Set context
        set_context(is_prefill=False, slot_mapping=slot_mapping,
                    context_lens=context_lens, block_tables=None)

        # Warmup
        hidden = self.model.model.embed_tokens(input_ids)
        residual = None
        for layer_id, layer in enumerate(self.model.model.layers):
            if layer_id % num_buffers == buffer_idx:
                hidden, residual = layer(positions, hidden, residual)

        # Capture
        with torch.cuda.graph(graph, self.offload_graph_pool):
            # Same operations
            ...

        self.offload_graphs[buffer_idx] = graph
```

#### Phase 3: Use Graphs in Decode

Modify `run_layerwise_offload_decode()` to replay graphs:

```python
for layer_id in range(num_layers):
    current_buffer = layer_id % num_buffers

    # Wait for H2D load
    offload_engine.wait_buffer_load(current_buffer)

    # Copy decode buffer to ring buffer (same as current)
    ...

    # Update graph variables
    self.offload_graph_vars["positions"][0] = positions[0]
    self.offload_graph_vars["slot_mapping"][0] = context_len
    self.offload_graph_vars["context_lens"][0] = context_len + 1

    # Replay graph instead of eager forward
    self.offload_graphs[current_buffer].replay()

    # Copy new KV to decode buffer (same as current)
    ...
```

### Challenges and Considerations

| Challenge | Solution |
|-----------|----------|
| H2D transfers interleaved with compute | H2D happens outside graph, only compute is captured |
| Different layers use different buffers | Capture per-buffer graphs, replay correct one |
| Variable context length | Use `cache_seqlens` parameter (fixed address, variable value) |
| Per-layer buffer rotation | Graph captures single-layer forward, loop in Python |

### Alternative: Full-Decode Graph (More Complex)

Instead of per-layer graphs, capture entire decode step:
1. Complete all H2D loads before graph
2. Single graph covers all layers
3. Better kernel fusion, less CPU overhead
4. More complex to implement (need to handle buffer rotation inside graph)

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Refactor offload decode to use Attention.forward() | ✅ Completed |
| Phase 1 | Implement `capture_offload_cudagraph()` with per-layer graphs | ✅ Completed |
| Phase 2 | Modify `run_layerwise_offload_decode()` to use graphs | ✅ Completed |
| Phase 3 | Test and benchmark | ✅ Completed |
| Phase 4 | (Optional) Optimize to full-decode graph | ⬜ Future |

## Architecture After Refactoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Offload Decode Flow (After Refactoring)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each layer:                                                            │
│    1. Wait for H2D load (ring buffer has prefill KV)                        │
│    2. Copy decode buffer → ring buffer (at prefill_len offset)              │
│    3. Set Attention.k_cache = ring_buffer[buffer_idx]                       │
│    4. Set context (slot_mapping, context_lens, block_tables=None)           │
│    5. layer.forward() → Qwen3Attention.forward() → Attention.forward()      │
│       └── store_kvcache() stores new token to ring buffer                   │
│       └── flash_attn_with_kvcache() computes attention                      │
│    6. Copy new token KV: ring buffer → decode buffer                        │
│    7. Start next layer H2D load                                             │
│                                                                             │
│  Key insight: Now uses standard Attention path, just with ring buffer       │
│  as k_cache/v_cache in contiguous format (block_tables=None)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files Modified

| File | Changes |
|------|---------|
| `model_runner.py:46-50` | Conditional CUDA graph capture: calls `capture_offload_cudagraph()` for offload mode |
| `model_runner.py:69-73` | Updated `exit()` to clean up offload graph resources |
| `model_runner.py:844-1031` | Refactored `run_layerwise_offload_decode()` to use standard `layer.forward()` with optional CUDA graph |
| `model_runner.py:1075-1164` | New `capture_offload_cudagraph()` method for per-layer graph capture |
| `tests/test_needle.py` | Added `--use-cuda-graph` flag to test CUDA graph mode |

## Implementation Details

### `capture_offload_cudagraph()` (line 1075-1164)

Captures per-layer CUDA graphs for offload decode:

```python
def capture_offload_cudagraph(self):
    # Fixed-address tensors for graph capture
    hidden_states = torch.randn(1, hidden_size, ...)
    residual = torch.randn(1, hidden_size, ...)
    layer_outputs = torch.zeros(1, hidden_size, ...)
    layer_residual = torch.zeros(1, hidden_size, ...)

    for layer_id in range(num_layers):
        buffer_idx = layer_id % num_buffers

        # Set Attention cache to ring buffer
        attn_module.k_cache = ring_buffer[buffer_idx:buffer_idx+1]
        attn_module.v_cache = ring_buffer[buffer_idx:buffer_idx+1]

        # Warmup and capture
        with torch.cuda.graph(graph):
            out_h, out_r = layer(positions, hidden_states, residual)
            layer_outputs.copy_(out_h)
            layer_residual.copy_(out_r)

        # Update inputs for next layer
        hidden_states.copy_(layer_outputs)
        residual.copy_(layer_residual)
```

### `run_layerwise_offload_decode()` CUDA Graph Mode

When CUDA graphs are available:

```python
use_cuda_graph = not self.enforce_eager and hasattr(self, 'offload_graphs')

if use_cuda_graph:
    # Use fixed-address tensors
    graph_vars["positions"][0] = len(seq) - 1
    graph_vars["slot_mapping"][0] = context_len
    graph_vars["context_lens"][0] = context_len + 1
    graph_vars["hidden_states"].copy_(embedding)
    graph_vars["residual"].zero_()

    for layer_id in range(num_layers):
        # Set up ring buffer and context
        ...

        # Replay graph
        self.offload_graphs[layer_id].replay()
        torch.cuda.current_stream().synchronize()

        # Copy outputs to inputs for next layer
        if layer_id < num_layers - 1:
            graph_vars["hidden_states"].copy_(graph_vars["layer_outputs"])
            graph_vars["residual"].copy_(graph_vars["layer_residual"])
```

## Test Results

| Test | Mode | CUDA Graph | Status |
|------|------|------------|--------|
| `test_needle.py --input-len 4096` | GPU-only | N/A | PASSED |
| `test_needle.py --input-len 4096 --enable-offload` | CPU offload | Disabled | PASSED |
| `test_needle.py --input-len 32768 --enable-offload` | CPU offload | Disabled | PASSED |
| `test_needle.py --input-len 32768 --enable-offload --use-cuda-graph` | CPU offload | Enabled | PASSED |

## Next Steps

1. ~~Implement `capture_offload_cudagraph()` method~~ ✅
2. ~~Modify `run_layerwise_offload_decode()` to optionally use captured graphs~~ ✅
3. ~~Test correctness with needle-in-haystack~~ ✅
4. Benchmark performance improvement from CUDA graphs (optional)
5. Consider full-decode graph optimization for maximum performance (future)
