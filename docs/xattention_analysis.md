# COMPASS XAttention Implementation Analysis

**Analysis Date**: 2026-01-14
**Researcher**: Claude Code Agent
**Source**: `/home/zijie/Code/COMPASS/compass/src/`

---

## Executive Summary

COMPASS XAttention is a **block sparse attention** implementation that uses:
1. **Approximation phase** (`xattn_estimate`) to compute attention importance and select blocks
2. **Computation phase** (`Xattention_prefill`) to compute sparse attention using `block_sparse_attn_func`
3. **Triton kernels** for efficient block-wise GEMM and softmax operations

**Key Integration Constraint**: Requires `block_sparse_attn_func` from flash-attention library, which is a **C++ CUDA extension** that must be compiled separately.

---

## 1. Function: `xattn_estimate()`

**Purpose**: Estimate attention importance and select which blocks to compute

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_states` | Tensor | - | Shape: `(batch, num_heads, q_len, head_dim)` |
| `key_states` | Tensor | - | Shape: `(batch, num_kv_heads, k_len, head_dim)` |
| `block_size` | int | - | Size of attention blocks (typically 128) |
| `stride` | int | - | Downsampling stride for approximation |
| `norm` | float | 1 | Normalization factor for attention scaling |
| `softmax` | bool | True | Whether to apply softmax in estimation |
| `threshold` | float | 0.9 | Block selection threshold (0-1) |
| `chunk_size` | int | 16384 | Processing chunk size |
| `select_mode` | str | "inverse" | Pattern selection mode |
| `use_triton` | bool | True | Use Triton kernels (requires SM 80+) |
| `causal` | bool | True | Apply causal masking |
| `kdb` | int | 1 | Key downsampling factor |
| `keep_sink` | bool | False | Always attend to first token |
| `keep_recent` | bool | False | Always attend to recent tokens |

### Output

```python
returns: (attn_sums, simple_masks)
  attn_sums: Tensor[float32]
    Shape: (batch, num_heads, num_q_blocks, num_k_blocks_per_chunk)
    Contains aggregated attention weights per block

  simple_masks: Tensor[bool]
    Shape: (batch, num_heads, num_q_blocks, num_k_blocks)
    Boolean mask indicating which blocks to compute
```

### Algorithm

#### Step 1: Padding and Chunking
```python
# Pad sequences to chunk_size boundaries
k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len

# Compute number of blocks and chunks
k_chunk_num = (k_len + k_num_to_pad) // chunk_size
k_block_num = (k_len + k_num_to_pad) // block_size
q_chunk_num = (q_len + q_num_to_pad) // chunk_size
q_block_num = (q_len + q_num_to_pad) // block_size
```

#### Step 2: Pattern Selection (stride-based downsampling)

**Purpose**: Reduce computation by `stride` factor using patterned selection

**Modes**:
1. **`"inverse"`** (default): Inverse stride pattern
   ```python
   # Key: regular stride [0, stride, 2*stride, ...]
   # Query: reverse stride [(stride-1), (stride-1-stride), ...]
   reshaped_key = torch.cat([key_states[:, :, k::stride, :] for k in range(stride)])
   reshaped_query = torch.cat([query_states[:, :, (stride-1-q)::stride*kdb, :] for q in range(stride)])
   ```

2. **`"slash"`**: Slash pattern (diagonal)
   ```python
   # Both use regular stride
   reshaped_key = torch.cat([key_states[:, :, k::stride, :] for k in range(stride)])
   reshaped_query = torch.cat([query_states[:, :, q::stride, :] for q in range(stride)])
   ```

3. **`"random"`**: Random permutation
4. **`"double"`, `"triple"`**: Data augmentation modes

#### Step 3: Chunk-wise Attention Estimation

For each query chunk:

**If `use_triton=True`** (fast path):
```python
# Triton kernel 1: Compute attention scores with fused reshape
attn_weights_slice = flat_group_gemm_fuse_reshape(
    query_chunk, key_states, stride,
    chunk_start, chunk_end, is_causal=causal
)

# Triton kernel 2: Softmax + block aggregation
attn_sum = softmax_fuse_block_sum(
    attn_weights_slice, reshaped_block_size, segment_size,
    chunk_start, chunk_end, real_q_len, scale, is_causal
)
```

**If `use_triton=False`** (PyTorch fallback):
```python
# Standard matrix multiplication
attn_weights_slice = torch.matmul(chunked_query, reshaped_key.transpose(2, 3))

# Scale and apply causal mask
attn_weights_slice = attn_weights_slice / sqrt(head_dim) / stride / norm
attn_weights_slice = attn_weights_slice + causal_mask

# Softmax
attn_weights_slice = F.softmax(attn_weights_slice, dim=-1)

# Aggregate to block level
attn_sum = attn_weights_slice.view(
    batch, heads, num_blocks_per_chunk, block_size//kdb, -1, block_size
).sum(dim=-1).sum(dim=-2)
```

#### Step 4: Block Selection

```python
# Select blocks based on threshold
simple_mask = find_blocks_chunked(
    attn_sum,
    current_index,  # Starting block index
    threshold,      # 0.9 = select blocks covering 90% of attention mass
    None,           # or num_to_choose for top-k selection
    decoding=False,
    mode="prefill",
    causal=True
)
```

**Selection Algorithm** (`find_blocks_chunked`):
1. Sort blocks by attention weight (descending)
2. Compute cumulative sum
3. Select blocks until `cumulative_sum >= total_sum * threshold`
4. Enforce causal constraints (no future blocks)
5. Always include sink token (first block) if `keep_sink=True`
6. Always include diagonal blocks if `keep_recent=True`

---

## 2. Function: `Xattention_prefill()`

**Purpose**: Compute sparse attention using estimated block mask

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_states` | Tensor | - | `(batch, num_heads, q_len, head_dim)` |
| `key_states` | Tensor | - | `(batch, num_heads, k_len, head_dim)` |
| `value_states` | Tensor | - | `(batch, num_heads, k_len, head_dim)` |
| `stride` | int | - | Downsampling stride for estimation |
| `norm` | float | 1 | Normalization factor |
| `threshold` | float | 0.8 | Block selection threshold |
| `block_size` | int | 128 | **MUST be 128** (hardcoded requirement) |
| `use_triton` | bool | True | Use Triton kernels in estimation |
| `causal` | bool | True | Apply causal masking |
| `kdb` | int | 1 | Key downsampling factor |
| `chunk_size` | int | None | Auto-computed if None |
| `keep_sink` | bool | False | Always attend to first token |
| `keep_recent` | bool | False | Always attend to recent tokens |

### Output

```python
returns: attn_output
  attn_output: Tensor
    Shape: (batch, num_heads, q_len, head_dim)
    Sparse attention output
```

### Algorithm Flow

#### Step 1: Auto-compute chunk_size
```python
if chunk_size is None:
    chunk_size = int(max(
        min(
            max(2048, 1 << (k_len - 1).bit_length()),  # Round to power of 2
            128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),  # Memory constraint
        ),
        2048,  # Minimum
    ))
```

**Example**:
- `k_len=8192` → `chunk_size=8192`
- `k_len=32768` → `chunk_size=16384`
- `k_len=65536` → `chunk_size=16384`

#### Step 2: Estimate attention and select blocks
```python
attn_sums, approx_simple_mask = xattn_estimate(
    query_states, key_states,
    block_size=block_size, stride=stride, norm=norm,
    threshold=threshold, select_mode="inverse",
    use_triton=use_triton, causal=causal,
    chunk_size=chunk_size, kdb=kdb,
    keep_sink=keep_sink, keep_recent=keep_recent
)
```

#### Step 3: Prepare inputs for block_sparse_attn_func
```python
# Hard constraints
assert block_size == 128
assert batch_size == 1

# Reshape to (seq_len, num_heads, head_dim)
query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)

# Cumulative sequence lengths
q_cu_seq_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
k_cu_seq_lens = torch.tensor([0, k_len], dtype=torch.int32, device=device)

# Head mask type (all heads use mask)
head_mask_type = torch.tensor([1 for _ in range(num_heads)], dtype=torch.int32)
```

#### Step 4: Call block_sparse_attn_func
```python
attn_output = block_sparse_attn_func(
    query_states,      # (q_len, num_heads, head_dim)
    key_states,        # (k_len, num_heads, head_dim)
    value_states,      # (k_len, num_heads, head_dim)
    q_cu_seq_lens,     # [0, q_len]
    k_cu_seq_lens,     # [0, k_len]
    head_mask_type,    # [1, 1, ..., 1]
    None,              # No custom layout
    approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),  # Block mask
    q_len,
    k_len,
    p_dropout=0.0,
    deterministic=True,
    is_causal=causal
)
```

#### Step 5: Reshape output
```python
attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)
# Output shape: (batch, num_heads, q_len, head_dim)
```

---

## 3. Triton Kernel Dependencies

### Kernel 1: `flat_group_gemm_fuse_reshape_kernel`

**Purpose**: Compute QK^T with stride-based reshaping

**Key Features**:
- Loads `stride` keys and queries at once
- Fused strided access pattern
- Causal masking support
- Block size auto-selection based on GPU memory

**Block Size Selection**:
```python
# RTX 3090 (<30GB): BLOCK_M=64, BLOCK_N=64
# A100/H100 (>=30GB): BLOCK_M=128, BLOCK_N=128
```

**Signature**:
```python
flat_group_gemm_fuse_reshape(
    query_states,  # (batch, heads, q_len, head_dim)
    key_states,    # (batch, heads, k_len, head_dim)
    stride,        # Downsampling factor
    chunk_start,   # Start position in keys
    chunk_end,     # End position in keys
    is_causal=True
)
# Returns: (batch, heads, q_len//stride, k_len//stride)
```

### Kernel 2: `softmax_fuse_block_sum_kernel_causal` / `_non_causal`

**Purpose**: Online softmax with block aggregation

**Algorithm**:
1. **Forward pass** (compute m_i, l_i):
   ```
   m_i = max(m_i, m_local)
   alpha = exp(m_i - m_new)
   l_i = l_i * alpha + sum(exp(X - m_new))
   ```
2. **Backward pass** (compute softmax with scaling):
   ```
   softmax = exp(X - m_i) / l_i
   aggregate to blocks: sum(softmax) over block_size
   ```

**Key Features**:
- Single-pass softmax (no materializing full attention matrix)
- Causal masking integrated
- Outputs block-level sums directly

**Signature**:
```python
softmax_fuse_block_sum(
    attn_weights_slice,  # (batch, heads, q_len, k_len)
    reshaped_block_size, # Block size (128//stride)
    segment_size,        # Processing segment (min(4096, block_size))
    chunk_start,         # Start position
    chunk_end,           # End position
    real_q_len,          # Actual query length (before padding)
    scale,               # 1.4426950408889634 / sqrt(head_dim) / stride / norm
    is_causal=True
)
# Returns: (batch, heads, q_len//block_size, k_len//block_size)
```

---

## 4. Key Parameters and Their Meanings

### Critical Parameters

| Parameter | Meaning | Typical Value | Impact |
|-----------|---------|---------------|--------|
| `block_size` | Block granularity | 128 | **Fixed at 128**, affects mask granularity |
| `stride` | Downsampling factor | 4-16 | Higher = faster but less accurate |
| `threshold` | Sparsity level | 0.8-0.9 | Higher = denser mask, more computation |
| `chunk_size` | Processing chunk | 16384 | Affects memory and efficiency |
| `kdb` | Key downsampling boost | 1 | Experimental, use 1 |
| `norm` | Scaling factor | 1.0 | Attention temperature control |

### Trade-offs

**Stride (`stride`)**:
- `stride=1`: No approximation, same as dense attention
- `stride=4`: 4x faster estimation, good accuracy
- `stride=8`: 8x faster, moderate accuracy loss
- `stride=16`: 16x faster, significant accuracy loss

**Threshold (`threshold`)**:
- `threshold=0.8`: Select blocks covering 80% of attention mass (~20% sparsity)
- `threshold=0.9`: Select blocks covering 90% of attention mass (~10% sparsity)
- `threshold=0.95`: Very dense, only prunes ~5% of blocks

---

## 5. Dependencies

### Required Libraries

1. **`block_sparse_attn`** (CRITICAL)
   - Source: `/home/zijie/Code/COMPASS/3rdparty/flash-attention/`
   - Function: `block_sparse_attn_func`
   - Type: **C++ CUDA extension**
   - Build: Requires compilation with `torch.utils.cpp_extension`

2. **Triton** (optional but recommended)
   - Required for: `use_triton=True`
   - GPU requirement: SM 80+ (A100, RTX 3090, H100, etc.)
   - Check: `torch.cuda.get_device_properties().major >= 8`

3. **PyTorch**
   - Version: Compatible with flash-attention
   - Features: F.pad, matmul, softmax, view, transpose

### Dependency Tree

```
Xattention_prefill
├── xattn_estimate
│   ├── flat_group_gemm_fuse_reshape (Triton)
│   ├── softmax_fuse_block_sum (Triton)
│   └── find_blocks_chunked (PyTorch)
└── block_sparse_attn_func (C++ CUDA)
```

---

## 6. Integration Issues for nano-vllm

### Critical Issue 1: `block_sparse_attn_func` Dependency

**Problem**: `block_sparse_attn_func` is a **C++ CUDA extension** that must be compiled from flash-attention source.

**Options**:
1. **Compile flash-attention with block sparse support**
   ```bash
   cd /home/zijie/Code/COMPASS/3rdparty/flash-attention
   python setup.py install
   ```
   - Risk: May conflict with existing flash-attention installation
   - Complexity: High (C++ compilation)

2. **Replace with FlashInfer block sparse**
   - FlashInfer is already a dependency
   - Has similar block sparse attention
   - Need to adapt interface

3. **Custom CUDA kernel**
   - Implement simplified block sparse attention
   - High development cost
   - Maintenance burden

### Critical Issue 2: Hard-coded Constraints

```python
assert block_size == 128  # Line 358
assert batch_size == 1    # Line 359
```

**Impact**:
- Cannot process multiple sequences in one batch
- Fixed block size limits flexibility
- Must work around these constraints

### Critical Issue 3: Triton GPU Requirement

```python
props = torch.cuda.get_device_properties(torch.cuda.current_device())
if props.major < 8:
    use_triton = False
```

**Impact**:
- Triton kernels only work on SM 80+ (A100, RTX 3090, H100)
- Older GPUs (V100, T4, RTX 2080) fall back to slow PyTorch implementation
- RTX 3090 works but uses smaller block sizes (64 vs 128)

### Issue 4: Memory Layout

**XAttention expects**:
```python
query_states: (batch, num_heads, q_len, head_dim)
```

**nano-vllm uses**:
```python
query_states: (num_heads, total_tokens, head_dim)  # Flattened batch
```

**Required**: Transpose and reshape before/after calling XAttention

### Issue 5: Chunking Incompatibility

**XAttention**: Processes in fixed-size chunks (e.g., 16384 tokens)
- Requires padding to chunk boundaries
- Adds overhead for short sequences

**nano-vllm**: Processes variable-length requests
- No padding requirement
- Dynamic batch sizing

---

## 7. Integration Strategy

### Recommended Approach: **Wrapper with FlashInfer**

1. **Keep `xattn_estimate`** (pure PyTorch + Triton)
   - No external dependencies
   - Computes block mask

2. **Replace `block_sparse_attn_func` with FlashInfer**
   - FlashInfer: `flashinfer.single_prefill_with_kv_cache`
   - Similar API, already compiled
   - Supports block sparse

3. **Adapt mask format**
   - XAttention: `(batch, heads, q_blocks, k_blocks)` boolean mask
   - FlashInfer: `(num_qo, num_kv)` boolean mask or custom format

4. **Handle constraints**
   - Enforce `batch_size=1` by processing one request at a time
   - Keep `block_size=128` as requirement

### Alternative: **Pure PyTorch Implementation**

1. Extract estimation algorithm
2. Implement sparse attention using PyTorch operations
3. Use FlashInfer for final computation
4. No Triton dependency

---

## 8. Code Example: Adaptation

```python
def xattention_prefill_adapted(
    query_states,  # (num_heads, q_len, head_dim)
    key_states,    # (num_heads, k_len, head_dim)
    value_states,  # (num_heads, k_len, head_dim)
    stride=4,
    threshold=0.9,
    block_size=128,
    causal=True,
):
    # Step 1: Add batch dimension
    q = query_states.unsqueeze(0)  # (1, heads, q_len, dim)
    k = key_states.unsqueeze(0)
    v = value_states.unsqueeze(0)

    # Step 2: Estimate mask (no external dependency)
    _, block_mask = xattn_estimate(
        q, k,
        block_size=block_size,
        stride=stride,
        threshold=threshold,
        use_triton=True,
        causal=causal,
    )
    # block_mask: (1, heads, q_blocks, k_blocks)

    # Step 3: Convert block mask to token mask
    q_blocks, k_blocks = block_mask.shape[-2:]
    token_mask = block_mask.repeat_interleave(block_size, dim=-2)
    token_mask = token_mask.repeat_interleave(block_size, dim=-1)
    token_mask = token_mask[:, :, :q.size(2), :k.size(2)]  # Trim padding

    # Step 4: Use FlashInfer with mask
    from flashinfer import single_prefill_with_kv_cache
    output = single_prefill_with_kv_cache(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        custom_mask=token_mask.squeeze(0),
    )

    return output  # (num_heads, q_len, head_dim)
```

---

## 9. Summary of Findings

### Advantages

1. **Accurate approximation**: Pattern-based stride selection preserves attention patterns
2. **Flexible sparsity**: Threshold-based control over computation
3. **GPU optimization**: Triton kernels for estimation phase
4. **Proven in practice**: Used in COMPASS system

### Challenges

1. **Hard dependency**: `block_sparse_attn_func` requires C++ compilation
2. **Rigid constraints**: `block_size=128`, `batch_size=1`
3. **GPU-specific**: Triton only on SM 80+
4. **Memory layout mismatch**: Requires reshape/transpose
5. **Chunking overhead**: Padding to chunk boundaries

### Integration Complexity

| Component | Complexity | Risk |
|-----------|------------|------|
| `xattn_estimate` | Medium | Low (PyTorch + Triton) |
| `block_sparse_attn_func` | High | **Critical** (C++ dependency) |
| Interface adaptation | Low | Low (reshape) |
| Constraint handling | Medium | Medium (workarounds) |

**Overall Integration Risk**: **HIGH** (due to C++ dependency)

---

## 10. Next Steps

1. **Evaluate FlashInfer compatibility**
   - Can FlashInfer replace `block_sparse_attn_func`?
   - What mask format does it expect?

2. **Prototype estimation phase**
   - Extract `xattn_estimate` function
   - Test with nano-vllm inputs
   - Validate mask quality

3. **Benchmark Triton kernels**
   - Compare Triton vs PyTorch estimation
   - Measure speedup on RTX 3090
   - Profile memory usage

4. **Design interface**
   - Define nano-vllm sparse attention API
   - Specify mask format
   - Plan integration points
