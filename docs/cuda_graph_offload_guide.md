# CUDA Graph Support for CPU Offload Mode

This document describes the CUDA graph implementation for the CPU offload decode path, which provides significant performance improvements for decode throughput.

## Overview

CUDA graphs capture a sequence of GPU operations and replay them with minimal CPU overhead. In offload mode, we capture per-layer graphs for the decode path, achieving **4x decode throughput improvement**.

## Performance Results

| Metric | Eager Mode | CUDA Graph | Improvement |
|--------|------------|------------|-------------|
| Decode Throughput | ~12 tok/s | ~50 tok/s | **4.2x** |
| TPOT (Time per output token) | ~80ms | ~19ms | **4.2x** |
| Prefill Throughput | ~8000 tok/s | ~8000 tok/s | Same |

## Architecture

### Why Standard CUDA Graph Capture Doesn't Work

The standard `capture_cudagraph()` captures the PagedAttention decode path:
- Uses block tables for scattered KV cache access
- `Attention.k_cache/v_cache` point to PagedAttention buffers

In offload mode, the decode path is different:
- Uses contiguous ring buffers for KV cache
- `Attention.k_cache/v_cache` dynamically point to ring buffer slices
- H2D transfers interleaved with compute

### Per-Layer Graph Design

We capture one CUDA graph per transformer layer:

```
┌─────────────────────────────────────────────────────────────┐
│                    Offload Decode with CUDA Graphs          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Initialization:                                            │
│    capture_offload_cudagraph() captures 36 layer graphs     │
│    Each graph: layer.forward() with ring buffer as cache    │
│                                                             │
│  Decode Step:                                               │
│    1. Embedding (eager, outside graph)                      │
│    2. For each layer:                                       │
│       a. Wait for H2D load (outside graph)                  │
│       b. Copy decode KV to ring buffer (outside graph)      │
│       c. Set Attention.k_cache = ring_buffer[buffer_idx]    │
│       d. Set context (slot_mapping, context_lens)           │
│       e. graph.replay() - layer forward                     │
│       f. synchronize()                                      │
│       g. Copy layer_outputs -> hidden_states                │
│       h. Copy new KV to decode buffer (outside graph)       │
│       i. Start next layer H2D load                          │
│    3. Final norm and logits (eager)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Ring Buffer Mapping

Each layer maps to a ring buffer slot:
```python
buffer_idx = layer_id % num_kv_buffers
```

With 4 buffers and 36 layers:
- Layer 0, 4, 8, ... use buffer 0
- Layer 1, 5, 9, ... use buffer 1
- Layer 2, 6, 10, ... use buffer 2
- Layer 3, 7, 11, ... use buffer 3

## Implementation Details

### Graph Capture (`capture_offload_cudagraph`)

Location: `model_runner.py:1075-1164`

```python
def capture_offload_cudagraph(self):
    # Fixed-address tensors for graph I/O
    hidden_states = torch.randn(1, hidden_size, ...)
    residual = torch.randn(1, hidden_size, ...)
    layer_outputs = torch.zeros(1, hidden_size, ...)
    layer_residual = torch.zeros(1, hidden_size, ...)

    for layer_id in range(num_layers):
        buffer_idx = layer_id % num_buffers

        # Set Attention cache to ring buffer slice
        attn_module.k_cache = ring_buffer[buffer_idx:buffer_idx+1]
        attn_module.v_cache = ring_buffer[buffer_idx:buffer_idx+1]

        # Set context for contiguous mode
        set_context(is_prefill=False, slot_mapping=...,
                    context_lens=..., block_tables=None)

        # Warmup and capture
        with torch.cuda.graph(graph, pool):
            out_h, out_r = layer(positions, hidden_states, residual)
            layer_outputs.copy_(out_h)
            layer_residual.copy_(out_r)

        # Propagate state for next layer's capture
        hidden_states.copy_(layer_outputs)
        residual.copy_(layer_residual)
```

Key design decisions:
1. **Fixed-address tensors**: Graph inputs/outputs use pre-allocated tensors
2. **Include copy in graph**: `layer_outputs.copy_(out_h)` is captured
3. **State propagation**: Update hidden_states between layer captures
4. **Random initialization**: Use `randn` instead of zeros for realistic distributions

### Graph Replay (`run_layerwise_offload_decode`)

Location: `model_runner.py:844-1031`

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
    # H2D and buffer setup (outside graph)
    offload_engine.wait_buffer_load(current_buffer)
    attn_module.k_cache = ring_buffer[current_buffer:current_buffer+1]
    set_context(...)

    if use_cuda_graph:
        # Replay graph
        self.offload_graphs[layer_id].replay()
        torch.cuda.current_stream().synchronize()

        # Copy outputs to inputs for next layer
        if layer_id < num_layers - 1:
            graph_vars["hidden_states"].copy_(graph_vars["layer_outputs"])
            graph_vars["residual"].copy_(graph_vars["layer_residual"])
    else:
        # Eager execution
        hidden_states, residual = layer(positions, hidden_states, residual)
```

Key points:
1. **Synchronization required**: `synchronize()` after each graph replay
2. **Manual state propagation**: Copy layer_outputs to hidden_states between replays
3. **H2D outside graph**: Ring buffer loads happen before graph replay

## Limitations and Future Work

### Current Limitations

1. **Per-layer sync overhead**: Each layer requires synchronization
2. **No kernel fusion across layers**: Each layer is a separate graph
3. **Fixed batch size**: Only supports batch_size=1 for offload

### Future Optimization: Full-Decode Graph

Potential improvement: Capture entire decode step as single graph
- Complete all H2D loads before graph
- Single graph covers all 36 layers
- Better kernel fusion, less CPU overhead
- More complex to implement (handle buffer rotation inside graph)

## Testing

Run needle test with CUDA graph:
```bash
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python tests/test_needle.py \
    --input-len 32768 \
    --enable-offload \
    --use-cuda-graph
```

Run benchmark:
```bash
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python bench_offload.py \
    --input-len 16384 \
    --bench-all
```

## Files Modified

| File | Changes |
|------|---------|
| `model_runner.py:46-50` | Call `capture_offload_cudagraph()` for offload mode |
| `model_runner.py:69-73` | Clean up offload graph resources in `exit()` |
| `model_runner.py:844-1031` | Add CUDA graph support to `run_layerwise_offload_decode()` |
| `model_runner.py:1075-1164` | New `capture_offload_cudagraph()` method |
| `tests/test_needle.py` | Add `--use-cuda-graph` flag |
