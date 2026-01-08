# Task Plan: Layerwise Offload Refactoring

## Goal
Refactor layerwise offload to use proper OffloadEngine API, pre-allocate buffers, remove chunked prefill code, and pass needle test.

## Phases
- [ ] Phase 1: Add layerwise API to OffloadEngine
- [ ] Phase 2: Pre-allocate buffers in ModelRunner
- [ ] Phase 3: Refactor run_layerwise_offload_prefill()
- [ ] Phase 4: Refactor run_layerwise_offload_decode()
- [ ] Phase 5: Remove chunked prefill code
- [ ] Phase 6: Verify with needle test

## Key Questions
1. Should we keep chunked_attention.py for MInference use?
2. What's the max_seq_len for buffer pre-allocation?
3. Should we implement incremental refactoring or all at once?

## Decisions Made
- Use FullAttentionPolicy for initial testing (per user request)
- Focus on correctness first, then optimize async overlap
- **GPU KV Cache使用Ring Buffer策略** (用户建议):
  - 使用N个buffer (可配置，默认4个) 形成ring buffer
  - 比固定2个buffer更灵活，流水线深度更深
  - 可以预加载多层，更好地隐藏H2D延迟
  - 例如: buffer[i] compute, buffer[(i+1)%N] load, buffer[(i+2)%N] load...

## Errors Encountered
(none yet)

## Status
**Currently in Phase 0** - Planning complete, awaiting user approval

---

## Detailed Implementation Plan

### Phase 1: Modify OffloadEngine GPU Memory Layout + Add Layerwise API

**File**: `nanovllm/kvcache/offload_engine.py`

#### 1.1 新的GPU内存布局 (Ring Buffer)

**设计原则**:
- 不追求极致的peak memory优化，而是保证流水线正确性和性能
- Ring buffer层数可从外部配置 (通过config或参数)
- 默认4层，可以根据GPU内存和H2D带宽调整

```python
# ========== Ring-Buffered GPU KV Cache for Layerwise Offload ==========
#
# 参数: num_kv_buffers (外部可配置，默认4)
#
# Ring Buffer流水线 (以4个buffer为例):
#   Buffer 0: [Load L0] → [Compute L0] ──────────────────────────► [Load L4]
#   Buffer 1:           [Load L1] → [Compute L1] ──────────────────────────►
#   Buffer 2:                     [Load L2] → [Compute L2] ────────────────►
#   Buffer 3:                               [Load L3] → [Compute L3] ──────►
#
# 优势:
# - 流水线深度 = num_kv_buffers - 1
# - 可以预加载多层，更好地隐藏H2D延迟
# - 比固定2层更灵活

def __init__(
    self,
    ...,
    num_kv_buffers: int = 4,  # 外部可配置的ring buffer层数
):
    self.num_kv_buffers = num_kv_buffers

    # Shape: [num_kv_buffers, max_seq_tokens, kv_heads, head_dim]
    self.layer_k_cache = torch.zeros(
        num_kv_buffers, max_seq_tokens, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    self.layer_v_cache = torch.zeros(
        num_kv_buffers, max_seq_tokens, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    # Per-buffer events for H2D completion
    self.buffer_load_events = [torch.cuda.Event() for _ in range(num_kv_buffers)]

# 内存开销计算 (Qwen3-4B, 128K tokens):
# - kv_heads=8, head_dim=128, dtype=bf16
# - 单层: 128K × 8 × 128 × 2 = 256 MB
# - 4层ring buffer: 4 × 256 MB = 1 GB
# - 对比28层全部在GPU: 28 × 256 MB = 7.2 GB
# - **节省**: 7.2 GB - 1 GB = 6.2 GB
```

**配置传递路径**:
```
LLM(num_kv_buffers=4)
  → Config.num_kv_buffers
    → OffloadEngine(num_kv_buffers=config.num_kv_buffers)
```

**移除旧的ring buffer设计**:
```python
# 移除: k_cache_gpu, v_cache_gpu (chunked prefill用的ring buffer)
# 移除: ring_slot_ready, ring_slot_offload_done, ring_slot_compute_done
# 移除: slot_transfer_streams
# 保留: prefill_offload_streams (用于D2H), compute_stream
```

#### 1.2 新的Layerwise API方法

```python
# ========== Prefill: Async D2H Offload ==========
def offload_layer_kv_async(
    self, layer_id: int, k: Tensor, v: Tensor,
    cpu_block_ids: list[int], total_tokens: int
) -> None:
    """Async offload layer KV to CPU using per-layer stream."""
    stream = self.prefill_offload_streams[layer_id]
    with torch.cuda.stream(stream):
        stream.wait_stream(self.compute_stream)  # Wait for compute
        for i, cpu_block_id in enumerate(cpu_block_ids):
            start = i * self.block_size
            end = min(start + self.block_size, total_tokens)
            self.k_cache_cpu[layer_id, cpu_block_id, :end-start].copy_(
                k[start:end], non_blocking=True
            )
            self.v_cache_cpu[layer_id, cpu_block_id, :end-start].copy_(
                v[start:end], non_blocking=True
            )
        self.prefill_offload_events[layer_id].record(stream)

def wait_layer_offload(self, layer_id: int) -> None:
    """Wait for specific layer's offload to complete."""
    self.compute_stream.wait_event(self.prefill_offload_events[layer_id])

# ========== Decode: Ring-Buffered H2D Load ==========
def load_layer_kv_to_buffer(
    self, buffer_idx: int, layer_id: int,
    cpu_block_ids: list[int], valid_tokens_per_block: list[int]
) -> None:
    """
    Async load layer KV from CPU to specified ring buffer slot.

    Args:
        buffer_idx: Ring buffer slot index (0 to num_kv_buffers-1)
        layer_id: Which layer's KV to load
        cpu_block_ids: CPU block IDs containing this layer's KV
        valid_tokens_per_block: Number of valid tokens in each block
    """
    stream = self.layer_load_streams[buffer_idx]  # 每个buffer有独立的stream
    with torch.cuda.stream(stream):
        # 等待该buffer上一次compute完成 (防止覆盖正在使用的数据)
        stream.wait_event(self.buffer_compute_done_events[buffer_idx])

        offset = 0
        for i, cpu_block_id in enumerate(cpu_block_ids):
            valid_tokens = valid_tokens_per_block[i]
            self.layer_k_cache[buffer_idx, offset:offset+valid_tokens].copy_(
                self.k_cache_cpu[layer_id, cpu_block_id, :valid_tokens],
                non_blocking=True
            )
            self.layer_v_cache[buffer_idx, offset:offset+valid_tokens].copy_(
                self.v_cache_cpu[layer_id, cpu_block_id, :valid_tokens],
                non_blocking=True
            )
            offset += valid_tokens
        self.buffer_load_events[buffer_idx].record(stream)

def wait_buffer_load(self, buffer_idx: int) -> None:
    """Wait for buffer load to complete on compute_stream."""
    self.compute_stream.wait_event(self.buffer_load_events[buffer_idx])

def get_buffer_kv(self, buffer_idx: int, total_tokens: int) -> tuple[Tensor, Tensor]:
    """Get KV from specified ring buffer slot."""
    return (
        self.layer_k_cache[buffer_idx, :total_tokens],
        self.layer_v_cache[buffer_idx, :total_tokens]
    )

def record_buffer_compute_done(self, buffer_idx: int) -> None:
    """Record that compute on this buffer is done (allows next load to reuse it)."""
    self.buffer_compute_done_events[buffer_idx].record(self.compute_stream)
```

#### 1.3 Ring Buffer所需的额外资源

```python
# Per-buffer streams (并行加载多个buffer)
self.layer_load_streams = [torch.cuda.Stream() for _ in range(num_kv_buffers)]

# Per-buffer events
self.buffer_load_events = [torch.cuda.Event() for _ in range(num_kv_buffers)]
self.buffer_compute_done_events = [torch.cuda.Event() for _ in range(num_kv_buffers)]

# 初始化: 标记所有buffer为"compute done" (允许首次加载)
for event in self.buffer_compute_done_events:
    event.record()
```

### Phase 2: Pre-allocate Buffers in ModelRunner

**File**: `nanovllm/engine/model_runner.py`

Add in `__init__()`:
```python
def _allocate_layerwise_buffers(self):
    max_seq_len = self.config.max_model_len
    hidden_size = self.config.hf_config.hidden_size
    num_heads = self.config.hf_config.num_attention_heads
    num_kv_heads = self.config.hf_config.num_key_value_heads
    head_dim = hidden_size // num_heads

    # QKV buffer for prefill
    self.prefill_qkv_buffer = torch.empty(
        max_seq_len, hidden_size + 2 * num_kv_heads * head_dim,
        dtype=self.dtype, device="cuda"
    )

    # Decode buffers (single token)
    self.decode_qkv_buffer = torch.empty(
        1, hidden_size + 2 * num_kv_heads * head_dim,
        dtype=self.dtype, device="cuda"
    )
```

### Phase 3: Refactor run_layerwise_offload_prefill()

**Key changes**:
1. Use `offload_engine.compute_stream` for all computation
2. Use `offload_layer_kv_async()` instead of `_offload_layer_kv_to_cpu_sync()`
3. Enable overlap: layer N offload overlaps with layer N+1 compute
4. Remove `torch.cuda.synchronize()`

```python
def run_layerwise_offload_prefill(self, seqs):
    offload_engine = self.kvcache_manager.offload_engine
    compute_stream = offload_engine.compute_stream

    with torch.cuda.stream(compute_stream):
        for layer_id in range(num_layers):
            # Wait for previous layer's offload buffer to be safe
            if layer_id > 0:
                offload_engine.wait_layer_offload(layer_id - 1)

            # Compute (using pre-allocated buffers where possible)
            q, k, v = compute_layer_qkv(...)
            attn_out = flash_attn_varlen_func(q, k, v, causal=True)
            hidden_states = compute_mlp(...)

            # Async offload (overlaps with next layer)
            offload_engine.offload_layer_kv_async(layer_id, k, v, cpu_block_ids, total_tokens)

    # Wait for final layer
    offload_engine.wait_layer_offload(num_layers - 1)
```

### Phase 4: Refactor run_layerwise_offload_decode()

**Key changes**:
1. 使用Ring Buffer实现compute/transfer overlap
2. N个buffer循环使用 (N = num_kv_buffers, 外部可配置)
3. 使用stream events而非global sync
4. 流水线深度 = N-1 (可预加载N-1层)

**Ring Buffer流水线示意** (以4个buffer为例):
```
时间 ────────────────────────────────────────────────────────────────────────►

Buffer 0: [Load L0] ─► [Compute L0] ────────────────────────► [Load L4] ─►
Buffer 1:            [Load L1] ─► [Compute L1] ────────────────────────►
Buffer 2:                       [Load L2] ─► [Compute L2] ────────────────►
Buffer 3:                                  [Load L3] ─► [Compute L3] ────►

流水线深度 = 3 (同时预加载3层)
```

```python
def run_layerwise_offload_decode(self, seqs):
    offload_engine = self.kvcache_manager.offload_engine
    compute_stream = offload_engine.compute_stream
    num_buffers = offload_engine.num_kv_buffers

    # 计算每个block的valid tokens
    valid_tokens_per_block = self._compute_valid_tokens(cpu_block_table, total_prefill_tokens)

    # Phase 1: 预加载前N层到ring buffer (填满流水线)
    num_preload = min(num_buffers, num_layers)
    for i in range(num_preload):
        offload_engine.load_layer_kv_to_buffer(
            i, i, cpu_block_table, valid_tokens_per_block
        )

    # Phase 2: 主循环 - compute当前层，load下一层
    with torch.cuda.stream(compute_stream):
        for layer_id in range(num_layers):
            # 1. 计算当前buffer index (ring)
            current_buffer = layer_id % num_buffers

            # 2. 等待当前buffer的加载完成
            offload_engine.wait_buffer_load(current_buffer)

            # 3. 开始加载下一层到同一buffer (buffer被复用)
            #    下一层 = layer_id + num_buffers (因为当前层用完后buffer可复用)
            next_layer_to_load = layer_id + num_buffers
            if next_layer_to_load < num_layers:
                offload_engine.load_layer_kv_to_buffer(
                    current_buffer, next_layer_to_load, cpu_block_table, valid_tokens_per_block
                )

            # 4. 获取当前buffer的KV并计算
            k_prefill, v_prefill = offload_engine.get_buffer_kv(current_buffer, total_prefill_tokens)

            # 5. 计算新token的QKV
            q_new, k_new, v_new = self._compute_decode_qkv(layer_id, hidden_states)

            # 6. 拼接并计算attention
            k_full = torch.cat([k_prefill, k_decode_prev, k_new], dim=0)
            v_full = torch.cat([v_prefill, v_decode_prev, v_new], dim=0)
            attn_out = flash_attn_varlen_func(q_new, k_full, v_full, causal=False)

            # 7. 标记当前buffer的compute完成 (允许后续load复用这个buffer)
            offload_engine.record_buffer_compute_done(current_buffer)

            # 8. 存储新KV到decode buffer
            offload_engine.decode_k_buffer[layer_id, pos].copy_(k_new.squeeze(0))
            offload_engine.decode_v_buffer[layer_id, pos].copy_(v_new.squeeze(0))

            # 9. MLP
            hidden_states = self._compute_mlp(layer_id, attn_out)

    # Block满时offload (使用async API)
    if block_is_full:
        offload_engine.offload_decode_buffer_async(cpu_block_id)
        # 注意: 这里不需要立即wait，可以在下一个decode step开始前wait
```

**优势**:
- Compute和H2D transfer完全overlap
- 流水线深度可配置 (num_kv_buffers-1)
- 没有global `torch.cuda.synchronize()`
- 使用stream events进行细粒度同步
- Buffer在layer_id + num_buffers时自动复用

### Phase 5: Remove Chunked Prefill Code

**Files to modify**:

| File | Remove |
|------|--------|
| `nanovllm/layers/attention.py` | `_chunked_prefill_attention()`, `_chunked_decode_attention()`, `_sync_load_previous_chunks()`, `_ring_buffer_pipeline_load()`, `_decode_ring_buffer_pipeline()`, `_decode_with_layer_pipeline()` |
| `nanovllm/utils/context.py` | `is_chunked_prefill`, `prev_kv_ranges`, `chunk_offset`, `chunked_seq`, `decode_pos_in_block`, `decode_start_pos_in_block`, `current_chunk_idx` |
| `nanovllm/kvcache/chunked_attention.py` | Keep for MInference (or remove if unused) |

Simplify `Attention.forward()` to:
```python
def forward(self, q, k, v):
    if context.is_prefill:
        if context.sparse_prefill_policy:
            return policy.sparse_prefill_attention(q, k, v, self.layer_id)
        else:
            return flash_attn_varlen_func(q, k, v, causal=True)
    else:
        return flash_attn_with_kvcache(q, k_cache, v_cache, causal=True)
```

### Phase 6: Verification

**Test command**:
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

**Success criteria**: `test_needle: PASSED`

---

## Current Issues Summary

| Issue | Location | Solution |
|-------|----------|----------|
| Direct `.copy_()` bypassing OffloadEngine | `model_runner.py:798-804` | Use `offload_layer_kv_async()` |
| `torch.cuda.synchronize()` | `model_runner.py:804` | Use stream events |
| Intermediate memory not pre-allocated | `model_runner.py:508-517` | Pre-allocate in `__init__()` |
| Chunked prefill code unused | `attention.py`, `context.py` | Remove entirely |

---

## Critical Files

- `nanovllm/kvcache/offload_engine.py` - Add layerwise API
- `nanovllm/engine/model_runner.py` - Pre-allocate buffers, refactor prefill/decode
- `nanovllm/layers/attention.py` - Remove chunked prefill code
- `nanovllm/utils/context.py` - Remove chunked prefill fields
