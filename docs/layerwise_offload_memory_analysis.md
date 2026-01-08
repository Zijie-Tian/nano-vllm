# Layer-wise Offload Memory Analysis

This document provides a detailed analysis of memory allocations in the layer-wise CPU offload system, distinguishing between pre-allocated (managed) memory and temporary (non-pre-allocated) memory.

## Variable Notation

| Symbol | Description | Example (Qwen3-4B) |
|--------|-------------|-------------------|
| `seq_len` | Input sequence length | 131072 (128k) |
| `hidden_size` | Model hidden dimension | 2560 |
| `num_heads` | Number of attention heads | 20 |
| `num_kv_heads` | Number of KV heads (GQA) | 8 |
| `head_dim` | Dimension per head | 128 |
| `intermediate_size` | MLP intermediate dimension | 13696 |
| `num_layers` | Number of transformer layers | 36 |
| `block_size` | KV cache block size | 1024 |
| `num_kv_buffers` | Ring buffer count | 4 |
| `num_cpu_blocks` | Number of CPU cache blocks | 128 |
| `vocab_size` | Vocabulary size | 151936 |
| `dtype_size` | Bytes per element (fp16/bf16) | 2 |

Derived values:
- `kv_dim = num_kv_heads × head_dim`
- `q_size = num_heads × head_dim`
- `kv_size = num_kv_heads × head_dim`
- `qkv_size = q_size + 2 × kv_size`

---

## 1. Pre-allocated Memory (Managed by nanovllm)

These tensors are allocated once during initialization and reused throughout inference.

### 1.1 OffloadEngine Managed Memory

| Tensor | Shape | Size Formula | Location |
|--------|-------|--------------|----------|
| `layer_k_cache` | `[num_kv_buffers, seq_len, num_kv_heads, head_dim]` | `num_kv_buffers × seq_len × kv_dim × dtype_size` | GPU |
| `layer_v_cache` | `[num_kv_buffers, seq_len, num_kv_heads, head_dim]` | `num_kv_buffers × seq_len × kv_dim × dtype_size` | GPU |
| `decode_k_buffer` | `[num_layers, block_size, num_kv_heads, head_dim]` | `num_layers × block_size × kv_dim × dtype_size` | GPU |
| `decode_v_buffer` | `[num_layers, block_size, num_kv_heads, head_dim]` | `num_layers × block_size × kv_dim × dtype_size` | GPU |
| `k_cache_cpu` | `[num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim]` | `num_layers × num_cpu_blocks × block_size × kv_dim × dtype_size` | CPU (pinned) |
| `v_cache_cpu` | `[num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim]` | `num_layers × num_cpu_blocks × block_size × kv_dim × dtype_size` | CPU (pinned) |

**Total GPU (OffloadEngine)**: `2 × (num_kv_buffers × seq_len + num_layers × block_size) × kv_dim × dtype_size`

**Total CPU (OffloadEngine)**: `2 × num_layers × num_cpu_blocks × block_size × kv_dim × dtype_size`

### 1.2 Model Weights

| Component | Approximate Size |
|-----------|-----------------|
| Embedding | `vocab_size × hidden_size × dtype_size` |
| Per-layer QKV proj | `hidden_size × qkv_size × dtype_size` |
| Per-layer O proj | `q_size × hidden_size × dtype_size` |
| Per-layer MLP | `hidden_size × 2 × intermediate_size × dtype_size + intermediate_size × hidden_size × dtype_size` |
| Per-layer LayerNorm | `2 × hidden_size × dtype_size` |
| LM Head | `hidden_size × vocab_size × dtype_size` |

### 1.3 RoPE Cache

| Tensor | Shape | Size |
|--------|-------|------|
| `cos_sin_cache` | `[max_position, 1, head_dim]` | `max_position × head_dim × 4` (float32) |

---

## 2. Non-Pre-allocated Memory: Prefill Phase

Location: `model_runner.py:run_layerwise_offload_prefill()`

### 2.1 Persistent Tensors (Live Throughout Prefill)

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `input_ids` | 488 | `[seq_len]` | `seq_len × 8` | int64 |
| `positions` | 489 | `[seq_len]` | `seq_len × 8` | int64 |
| `cu_seqlens` | 493 | `[2]` | negligible | int32 |
| `hidden_states` | 497 | `[seq_len, hidden_size]` | `seq_len × hidden_size × dtype_size` | Embedding output |
| `residual` | 506 | `[seq_len, hidden_size]` | `seq_len × hidden_size × dtype_size` | Residual connection |

### 2.2 Per-Layer Temporary Tensors

These are allocated and deallocated within each layer iteration.

#### 2.2.1 LayerNorm

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `hidden_ln` | 506-508 | `[seq_len, hidden_size]` | `seq_len × hidden_size × dtype_size` | Input layernorm output |

**Inside RMSNorm** (`layernorm.py:add_rms_forward`):
| Variable | Shape | Size | Notes |
|----------|-------|------|-------|
| `x.float()` | `[seq_len, hidden_size]` | `seq_len × hidden_size × 4` | Upcasted to float32 |
| `var` | `[seq_len, 1]` | `seq_len × 4` | Variance |

#### 2.2.2 QKV Projection

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `qkv` | 512 | `[seq_len, q_size + 2 × kv_size]` | `seq_len × qkv_size × dtype_size` | Merged QKV output |
| `q` | 513-519 | `[seq_len, num_heads, head_dim]` | 0 (view) | View of qkv |
| `k` | 513-520 | `[seq_len, num_kv_heads, head_dim]` | 0 (view) | View of qkv |
| `v` | 513-521 | `[seq_len, num_kv_heads, head_dim]` | 0 (view) | View of qkv |

#### 2.2.3 Q/K Norms (Qwen3 specific)

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `q.reshape()` | 526 | `[seq_len × num_heads, head_dim]` | 0 (view) | Reshape for norm |
| `k.reshape()` | 528 | `[seq_len × num_kv_heads, head_dim]` | 0 (view) | Reshape for norm |
| RMSNorm intermediates | - | see above | `seq_len × num_heads × head_dim × 4` | Float32 upcasting |

#### 2.2.4 RoPE (Rotary Position Embedding)

Location: `rotary_embedding.py:apply_rotary_emb()`

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `cos_sin` | 44 | `[seq_len, 1, head_dim]` | 0 (view) | View of cached cos_sin |
| `cos` | 45 | `[seq_len, 1, head_dim/2]` | 0 (view) | Chunk view |
| `sin` | 45 | `[seq_len, 1, head_dim/2]` | 0 (view) | Chunk view |

**Inside `apply_rotary_emb` for Q** (`rotary_embedding.py:6-14`):
| Variable | Shape | Size | Notes |
|----------|-------|------|-------|
| `x.float()` | `[seq_len, num_heads, head_dim]` | `seq_len × num_heads × head_dim × 4` | Upcast to float32 |
| `x1` | `[seq_len, num_heads, head_dim/2]` | 0 (view) | Chunk view |
| `x2` | `[seq_len, num_heads, head_dim/2]` | 0 (view) | Chunk view |
| `y1 = x1*cos - x2*sin` | `[seq_len, num_heads, head_dim/2]` | `seq_len × num_heads × head_dim/2 × 4` | New tensor |
| `y2 = x2*cos + x1*sin` | `[seq_len, num_heads, head_dim/2]` | `seq_len × num_heads × head_dim/2 × 4` | New tensor |
| `torch.cat((y1, y2))` | `[seq_len, num_heads, head_dim]` | `seq_len × num_heads × head_dim × 4` | New tensor |
| `.to(x.dtype)` | `[seq_len, num_heads, head_dim]` | `seq_len × num_heads × head_dim × dtype_size` | Downcast |

**Inside `apply_rotary_emb` for K**:
| Variable | Shape | Size | Notes |
|----------|-------|------|-------|
| Same pattern as Q | `[seq_len, num_kv_heads, head_dim]` | Similar, with `num_kv_heads` | |

**Total RoPE temporary for Q+K**: ~`seq_len × (num_heads + num_kv_heads) × head_dim × 4 × 3` (float32 intermediates)

#### 2.2.5 FlashAttention

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `attn_output` | 535 | `[seq_len, num_heads, head_dim]` | `seq_len × num_heads × head_dim × dtype_size` | Attention output |
| Internal workspace | - | O(seq_len) | Variable | FlashAttention internal |

#### 2.2.6 Output Projection

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `attn_output.view()` | 546 | `[seq_len, q_size]` | 0 (view) | Reshape for o_proj |
| `o_proj(attn_output)` | 547 | `[seq_len, hidden_size]` | `seq_len × hidden_size × dtype_size` | O projection output |

#### 2.2.7 Post-Attention LayerNorm

Same as input layernorm (2.2.1).

#### 2.2.8 MLP

Location: `qwen3.py:Qwen3MLP.forward()`

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `gate_up` | 117 | `[seq_len, 2 × intermediate_size]` | `seq_len × 2 × intermediate_size × dtype_size` | **LARGEST TEMPORARY!** |
| `x, y = chunk()` | activation.py:13 | `[seq_len, intermediate_size]` × 2 | 0 (views) | Chunk views |
| `F.silu(x)` | activation.py:14 | `[seq_len, intermediate_size]` | `seq_len × intermediate_size × dtype_size` | SiLU activation |
| `silu(x) * y` | activation.py:14 | `[seq_len, intermediate_size]` | `seq_len × intermediate_size × dtype_size` | Gated output |
| `down_proj()` | 119 | `[seq_len, hidden_size]` | `seq_len × hidden_size × dtype_size` | MLP output |

### 2.3 Prefill Memory Summary

**Peak per-layer temporary memory**:
```
= qkv + RoPE_temps + attn_output + o_proj + layernorm + MLP_gate_up + MLP_activation
≈ seq_len × (qkv_size + (num_heads + num_kv_heads) × head_dim × 4 × 3
           + num_heads × head_dim + hidden_size × 2 + 2 × intermediate_size + intermediate_size) × dtype_size
```

**Dominant term**: `seq_len × 2 × intermediate_size × dtype_size` (MLP gate_up)

---

## 3. Non-Pre-allocated Memory: Decode Phase

Location: `model_runner.py:run_layerwise_offload_decode()`

### 3.1 Persistent Tensors

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `input_ids` | 604 | `[1]` | 8 bytes | Single token |
| `positions` | 605 | `[1]` | 8 bytes | Single position |
| `cu_seqlens_q` | 631 | `[2]` | 8 bytes | Fixed |
| `valid_tokens_per_block` | 613-622 | Python list | negligible | |

### 3.2 Per-Layer Temporary Tensors

#### 3.2.1 Views (Zero Additional Memory)

| Variable | Line | Shape | Notes |
|----------|------|-------|-------|
| `k_prefill` | 682 | `[prefill_len, num_kv_heads, head_dim]` | View of ring buffer |
| `v_prefill` | 682 | `[prefill_len, num_kv_heads, head_dim]` | View of ring buffer |
| `k_decode_prev` | 686-687 | `[num_decode_tokens-1, num_kv_heads, head_dim]` | View of decode buffer |
| `v_decode_prev` | 686-688 | `[num_decode_tokens-1, num_kv_heads, head_dim]` | View of decode buffer |

#### 3.2.2 New Allocations

| Variable | Line | Shape | Size | Notes |
|----------|------|-------|------|-------|
| `hidden_ln` | 654-657 | `[1, hidden_size]` | `hidden_size × dtype_size` | Tiny |
| `qkv` | 660 | `[1, qkv_size]` | `qkv_size × dtype_size` | Tiny |
| `q` | 667 | `[1, num_heads, head_dim]` | 0 (view) | |
| `k_new` | 668 | `[1, num_kv_heads, head_dim]` | 0 (view) | |
| `v_new` | 669 | `[1, num_kv_heads, head_dim]` | 0 (view) | |
| **`k_full`** | 689/692 | `[prefill_len + num_decode_tokens, num_kv_heads, head_dim]` | `(prefill_len + num_decode_tokens) × kv_dim × dtype_size` | **torch.cat - NEW ALLOCATION** |
| **`v_full`** | 690/693 | `[prefill_len + num_decode_tokens, num_kv_heads, head_dim]` | `(prefill_len + num_decode_tokens) × kv_dim × dtype_size` | **torch.cat - NEW ALLOCATION** |
| `cu_seqlens_k` | 710 | `[2]` | 8 bytes | Created per layer |
| `attn_output` | 712 | `[1, num_heads, head_dim]` | `num_heads × head_dim × dtype_size` | Tiny |
| MLP temps | 728 | `[1, ...]` | negligible | Single token |

### 3.3 Decode Memory Summary

**Peak per-layer temporary memory**:
```
= k_full + v_full + small_tensors
≈ 2 × (prefill_len + num_decode_tokens) × num_kv_heads × head_dim × dtype_size
≈ 2 × seq_len × kv_dim × dtype_size
```

**Dominant term**: `k_full` and `v_full` from `torch.cat()`

---

## 4. Memory Comparison Table

For Qwen3-4B with 128k context:

| Category | Memory | Notes |
|----------|--------|-------|
| **Pre-allocated GPU** | ~2.2 GB | Ring buffer + decode buffer |
| **Pre-allocated CPU** | ~18.4 GB | Pinned memory |
| **Model Weights** | ~8 GB | |
| **Prefill Peak Temp** | ~10-12 GB | MLP gate_up dominant |
| **Decode Peak Temp** | ~512 MB | k_full + v_full |

---

## 5. Optimization Opportunities

### 5.1 Decode: Pre-allocate k_full/v_full

**Current** (L689-693):
```python
k_full = torch.cat([k_prefill, k_decode_prev, k_new], dim=0)  # New allocation each layer
v_full = torch.cat([v_prefill, v_decode_prev, v_new], dim=0)  # New allocation each layer
```

**Optimized**:
```python
# Pre-allocate in OffloadEngine.__init__():
self.k_full_buffer = torch.zeros(max_seq_len + block_size, num_kv_heads, head_dim, ...)
self.v_full_buffer = torch.zeros(max_seq_len + block_size, num_kv_heads, head_dim, ...)

# In decode loop:
total_len = prefill_len + num_decode_tokens
k_full = self.k_full_buffer[:total_len]
k_full[:prefill_len].copy_(k_prefill)
k_full[prefill_len:prefill_len+num_decode_prev].copy_(k_decode_prev)
k_full[-1:].copy_(k_new)
```

**Savings**: ~512 MB per decode step (for 128k)

### 5.2 Decode: Reuse cu_seqlens_k

**Current** (L710):
```python
cu_seqlens_k = torch.tensor([0, total_kv_tokens], dtype=torch.int32, device="cuda")
```

**Optimized**:
```python
# Pre-allocate once:
self.cu_seqlens_k = torch.zeros(2, dtype=torch.int32, device="cuda")

# In decode loop:
self.cu_seqlens_k[1] = total_kv_tokens
```

**Savings**: Negligible memory, but reduces allocation overhead.

### 5.3 RoPE: In-place or Pre-allocated Buffers

The RoPE implementation creates multiple float32 intermediate tensors. Options:
1. Pre-allocate buffers for Q and K rotary outputs
2. Use in-place operations where possible
3. Use fused RoPE kernel (e.g., from FlashAttention)

**Potential savings**: ~1.5 GB during prefill per layer

### 5.4 MLP: Cannot Optimize Easily

The MLP `gate_up` tensor is inherently required for the gated activation:
```python
gate_up = gate_up_proj(x)  # [seq_len, 2 × intermediate_size]
x, y = gate_up.chunk(2, -1)
output = silu(x) * y
```

This is a fundamental computation pattern. Potential optimizations:
- Chunked MLP computation (process seq_len in chunks)
- Fused kernels that avoid materializing full gate_up

---

## 6. Memory Flow Diagram

### Prefill (per layer):

```
hidden_states ──┬──► LayerNorm ──► hidden_ln
                │
residual ◄──────┘

hidden_ln ──► QKV_proj ──► qkv ──┬──► q ──► Q_norm ──► RoPE ──► q_rotated
                                 ├──► k ──► K_norm ──► RoPE ──► k_rotated
                                 └──► v

q_rotated, k_rotated, v ──► FlashAttention ──► attn_output

attn_output ──► O_proj ──► hidden_states'

hidden_states', residual ──► LayerNorm ──► hidden_ln', residual'

hidden_ln' ──► MLP_gate_up ──► gate_up ──► SiLU×gate ──► MLP_down ──► hidden_states''

k_rotated, v ──► CPU_offload (sync copy)
```

### Decode (per layer):

```
[CPU] k_cache_cpu, v_cache_cpu
           │
           ▼ (H2D async to ring buffer)
[GPU] layer_k_cache[buffer_idx], layer_v_cache[buffer_idx]
           │
           ▼ (view)
      k_prefill, v_prefill
           │
           ├──► torch.cat([k_prefill, k_decode_prev, k_new]) ──► k_full  ⚠️ NEW ALLOC
           │
           └──► torch.cat([v_prefill, v_decode_prev, v_new]) ──► v_full  ⚠️ NEW ALLOC

q_new, k_full, v_full ──► FlashAttention ──► attn_output

k_new, v_new ──► decode_k_buffer, decode_v_buffer (in-place store)
```

---

## 7. Appendix: Size Calculations

### Qwen3-4B Example (128k context)

```python
# Model config
seq_len = 131072
hidden_size = 2560
num_heads = 20
num_kv_heads = 8
head_dim = 128
intermediate_size = 13696
num_layers = 36
block_size = 1024
num_kv_buffers = 4
num_cpu_blocks = 128
dtype_size = 2  # fp16/bf16

# Derived
kv_dim = num_kv_heads * head_dim  # 1024
q_size = num_heads * head_dim     # 2560
qkv_size = q_size + 2 * kv_dim    # 4608

# Pre-allocated GPU (OffloadEngine)
ring_buffer = 2 * num_kv_buffers * seq_len * kv_dim * dtype_size
# = 2 * 4 * 131072 * 1024 * 2 = 2,147,483,648 bytes = 2048 MB

decode_buffer = 2 * num_layers * block_size * kv_dim * dtype_size
# = 2 * 36 * 1024 * 1024 * 2 = 150,994,944 bytes = 144 MB

# Pre-allocated CPU
cpu_cache = 2 * num_layers * num_cpu_blocks * block_size * kv_dim * dtype_size
# = 2 * 36 * 128 * 1024 * 1024 * 2 = 19,327,352,832 bytes = 18432 MB

# Prefill temporaries (per layer peak)
mlp_gate_up = seq_len * 2 * intermediate_size * dtype_size
# = 131072 * 2 * 13696 * 2 = 7,180,648,448 bytes = 6848 MB

# Decode temporaries (per layer)
k_full = seq_len * kv_dim * dtype_size
# = 131072 * 1024 * 2 = 268,435,456 bytes = 256 MB
v_full = k_full  # = 256 MB
# Total: 512 MB
```
