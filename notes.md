# Notes: Layerwise Offload Implementation

## Code Analysis

### Current Layerwise Offload Flow

**Prefill** (`model_runner.py:462-573`):
```
for layer_id in range(num_layers):
    q, k, v = compute_qkv(hidden_states)
    attn_out = flash_attn_varlen_func(q, k, v, causal=True)
    hidden_states = mlp(attn_out)
    _offload_layer_kv_to_cpu_sync(layer_id, k, v)  # BLOCKING!
```

**Decode** (`model_runner.py:641-817`):
```
for layer_id in range(num_layers):
    # Load all prefilled KV from CPU (SLOW!)
    for block_id in cpu_block_table:
        k_block = k_cache_cpu[layer_id, block_id].to("cuda")
        v_block = v_cache_cpu[layer_id, block_id].to("cuda")

    k_full = cat([k_prefill, k_decode_prev, k_new])
    attn_out = flash_attn(q, k_full, v_full, causal=False)

    # Store new KV to decode buffer
    decode_k_buffer[layer_id, pos].copy_(k_new)

# Block-full offload (lines 793-811)
if block_is_full:
    for layer_id in range(num_layers):
        k_cache_cpu[layer_id, block].copy_(decode_k_buffer[layer_id], non_blocking=True)
    torch.cuda.synchronize()  # BAD: global sync
```

### OffloadEngine Existing Infrastructure

**Streams** (available for use):
- `compute_stream` - dedicated compute stream (not default!)
- `prefill_offload_streams[layer_id]` - per-layer D2H streams
- `slot_transfer_streams[slot_idx]` - per-slot H2D streams
- `transfer_stream_main` - main transfer stream
- `_pipeline_layer_stream` - cross-layer pipeline stream

**Events** (available for use):
- `prefill_offload_events[layer_id]` - per-layer offload completion
- `ring_slot_ready[slot]` - H2D completion
- `ring_slot_offload_done[slot]` - D2H completion
- `ring_slot_compute_done[slot]` - compute completion
- `_pipeline_next_layer_event` - pipeline next layer ready

**Buffers** (already allocated):
- `k_cache_cpu/v_cache_cpu` - [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
- `k_cache_gpu/v_cache_gpu` - [num_gpu_blocks, block_size, kv_heads, head_dim] (no layer dim!)
- `decode_k_buffer/v_buffer` - [num_layers, block_size, kv_heads, head_dim]
- `prefill_k_buffer/v_buffer` - [num_layers, block_size, kv_heads, head_dim]
- `layer_k_buffer_a/b, layer_v_buffer_a/b` - [max_prefill_blocks, block_size, kv_heads, head_dim]

### Useful Existing Methods

**Async offload** (currently unused in layerwise):
```python
offload_prefill_buffer_async(layer_id, cpu_block_id, num_valid_tokens)
wait_all_prefill_offloads()
wait_prefill_offload(layer_id)
```

**Cross-layer pipeline** (for decode):
```python
start_decode_pipeline(cpu_block_ids)
get_decode_layer_kv(layer_id, num_blocks) -> (k, v)
end_decode_pipeline()
```

### Chunked Prefill Code to Remove

**attention.py** (lines to remove):
- 172-312: `_chunked_prefill_attention()`
- 314-346: `_sync_load_previous_chunks()`
- 348-480: `_ring_buffer_pipeline_load()`
- 482-591: `_chunked_decode_attention()`
- 593-667: `_decode_ring_buffer_pipeline()`
- 669-726: `_decode_with_layer_pipeline()`

**context.py** (fields to remove):
- `is_chunked_prefill`
- `prev_kv_ranges`
- `chunk_offset`
- `chunked_seq`
- `decode_pos_in_block`
- `decode_start_pos_in_block`
- `current_chunk_idx`

**Keep**:
- `kvcache_manager` - still needed for layerwise
- `sparse_prefill_policy` - needed for MInference

---

## Memory Layout

### 新设计: Ring-Buffered GPU KV Cache

**设计原则**:
- 不追求极致peak memory优化，保证流水线正确性
- Ring buffer层数可从外部配置 (默认4层)
- 流水线深度 = num_kv_buffers - 1

```
# 新: Ring-Buffered GPU Cache (layerwise offload专用)
# num_kv_buffers: 外部可配置，默认4
layer_k_cache: [num_kv_buffers, max_seq_tokens, kv_heads, head_dim]
layer_v_cache: [num_kv_buffers, max_seq_tokens, kv_heads, head_dim]

# 移除: 旧的chunked prefill ring buffer
# k_cache_gpu: [num_gpu_blocks, block_size, kv_heads, head_dim]  <- 删除
# v_cache_gpu: [num_gpu_blocks, block_size, kv_heads, head_dim]  <- 删除
```

**为什么使用Ring Buffer?**

Decode阶段的流水线需求 (以4个buffer为例):
```
Buffer 0: [Load L0] → [Compute L0] ──────────────────► [Load L4]
Buffer 1:           [Load L1] → [Compute L1] ────────────────────►
Buffer 2:                     [Load L2] → [Compute L2] ────────────►
Buffer 3:                               [Load L3] → [Compute L3] ──►
```

流水线深度 = 3，可以预加载3层，更好地隐藏H2D延迟。

**内存开销** (Qwen3-4B, 128K tokens):
- 单层KV: 128K × 8 × 128 × 2 bytes = 256 MB
- 4层ring buffer: 4 × 256 MB = 1 GB
- 对比28层全GPU: 28 × 256 MB = 7.2 GB
- **节省**: 7.2 GB - 1 GB = 6.2 GB

**配置传递**:
```
LLM(num_kv_buffers=4) → Config → OffloadEngine(num_kv_buffers=...)
```

### CPU Cache (保持不变)
```
k_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
v_cache_cpu: [num_layers, num_cpu_blocks, block_size, kv_heads, head_dim]
```
Pinned memory for fast DMA transfers.

### Memory per Layer (Qwen3-4B)
- kv_heads = 8
- head_dim = 128
- dtype = bfloat16 (2 bytes)
- Per token KV: 8 * 128 * 2 * 2 = 4KB
- 128K tokens: 512 MB per layer
- 28 layers: 14 GB total on CPU

---

## Stream Synchronization Pattern

### Correct Pattern for Async Offload
```python
# In offload stream
with torch.cuda.stream(offload_stream):
    offload_stream.wait_stream(compute_stream)  # Wait for compute to finish
    cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    event.record(offload_stream)

# Before reusing gpu_tensor
compute_stream.wait_event(event)  # Wait for offload to complete
```

### Correct Pattern for Async Load
```python
# In load stream
with torch.cuda.stream(load_stream):
    gpu_buffer.copy_(cpu_tensor, non_blocking=True)
    event.record(load_stream)

# Before using gpu_buffer
compute_stream.wait_event(event)  # Wait for load to complete
```

---

## Test Configuration

**Needle test command**:
```bash
PYTHONPATH=/home/zijie/.claude-squad/worktrees/zijie/int-offload-1_188890c8699249f7:$PYTHONPATH \
python tests/test_needle.py \
    --model ~/models/Qwen3-4B-Instruct-2507/ \
    --max-model-len 32768 \
    --input-len 8192 \
    --enable-offload \
    --block-size 1024 \
    --num-gpu-blocks 2
```

**GPU mutex check before running**:
```bash
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```
