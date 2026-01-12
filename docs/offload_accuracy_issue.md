# CPU Offload Accuracy Issue Investigation

## Problem Summary

CPU offload mode produces significantly lower accuracy than non-offload mode on the RULER NIAH benchmark.

| Mode | Accuracy | Pass/Total |
|------|----------|------------|
| **Non-Offload (GPU only)** | **100%** | 100/100 |
| **CPU Offload** | **66%** | 66/100 |

This 34% accuracy drop indicates a bug in the offload implementation that affects inference correctness.

## Test Environment

- **Model**: Llama-3.1-8B-Instruct
- **Task**: RULER NIAH (Needle-In-A-Haystack) 32K context
- **GPU**: NVIDIA A100-SXM4-80GB
- **Data**: `tests/data/ruler_niah/niah_single_1_32k.jsonl` (100 samples)

## Reproduction Commands

### Non-Offload Mode (100% accuracy)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --gpu-utilization 0.7 \
    --quiet
```

**Configuration**:
- KV Cache: GPU only, 51 blocks (6528 MB)
- Block size: 1024 tokens

### Offload Mode (66% accuracy)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --quiet
```

**Configuration**:
- KV Cache: GPU 4 blocks (512 MB) + CPU 32 blocks (4096 MB)
- Ring buffer: 4 buffers × 33280 tokens (520 MB)
- Per-layer decode buffer: 128 MB
- Block size: 1024 tokens

## Observed Failure Patterns

From the 5-sample verbose test:

| Sample | Expected | Offload Output | Status |
|--------|----------|----------------|--------|
| 0 | 8930103 | `: 8930103.` | PASS |
| 1 | 4194548 | `: 419 multiplication of 4548.` | **FAIL** |
| 2 | 8231838 | `:ное 8231838.` | PASS |
| 3 | 8835373 | `: 8835373.` | PASS |
| 4 | 7754864 | `aster 7754864.` | PASS |

**Failure pattern**: The model sometimes produces corrupted or split outputs (e.g., "419 multiplication of 4548" instead of "4194548").

## Architecture Overview

### Offload Mode Data Flow

```
Prefill Phase:
1. Input tokens → chunked into 2048-token chunks
2. Each chunk processed layer by layer:
   - Load KV from CPU → GPU ring buffer
   - Compute attention
   - Store KV back to CPU
3. Ring buffer holds recent KV for decode

Decode Phase:
1. For each new token:
   - Load all layer KV from CPU (one layer at a time)
   - Compute attention against full context
   - Generate next token
```

### Key Components

| File | Component | Description |
|------|-----------|-------------|
| `nanovllm/kvcache/offload_engine.py` | `OffloadEngine` | Manages CPU↔GPU KV cache transfers |
| `nanovllm/kvcache/offload_engine.py` | `RingKVBuffer` | GPU ring buffer for recent KV |
| `nanovllm/engine/model_runner.py` | `run_chunked_offload_prefill()` | Chunked prefill with offload |
| `nanovllm/engine/model_runner.py` | `run_offload_decode()` | Layer-wise decode with offload |
| `nanovllm/kvcache/hybrid_manager.py` | `HybridBlockManager` | CPU block allocation |

## Potential Root Causes

### 1. Ring Buffer Index/Position Issues

**Location**: `nanovllm/kvcache/offload_engine.py`

The ring buffer uses modular indexing. Potential issues:
- Position calculation errors during prefill/decode transition
- Off-by-one errors in KV storage/retrieval
- Incorrect handling when sequence length approaches `max_seq_len`

**Recent fix applied**: `max_seq_len = max_model_len + 512` to prevent overflow, but there may be other indexing issues.

### 2. Chunked Prefill KV Storage

**Location**: `nanovllm/engine/model_runner.py:run_chunked_offload_prefill()`

During chunked prefill:
- KV computed for chunk N must be correctly stored before processing chunk N+1
- Position IDs must be correctly accumulated across chunks
- CPU block allocation must be contiguous and correctly tracked

**Suspect areas**:
```python
# Check if positions are correctly tracked across chunks
# Check if KV is correctly copied to CPU after each chunk
# Check if ring buffer indices align with CPU block indices
```

### 3. Decode Phase KV Loading

**Location**: `nanovllm/engine/model_runner.py:run_offload_decode()`

During decode:
- Must load KV for ALL previous tokens (both prefill and decode)
- Layer-by-layer loading must be synchronized correctly
- Attention computation must use correct sequence length

**Suspect areas**:
```python
# Check if decode loads KV for full context length
# Check if new decode KV is stored correctly
# Check if attention mask/positions are correct
```

### 4. CPU↔GPU Transfer Synchronization

**Location**: `nanovllm/kvcache/offload_engine.py`

CUDA streams and synchronization:
- Async copies may complete out of order
- Missing synchronization points could cause stale data
- Stream priorities may affect correctness

### 5. Numerical Precision

- CPU tensors use float16/bfloat16
- GPU computation precision
- Potential precision loss during transfers

## Debugging Strategy

### Step 1: Identify Failing Samples

```bash
# Run verbose mode to see which samples fail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --verbose 2>&1 | tee offload_verbose.log
```

### Step 2: Compare Token-by-Token

Create a debug script to compare token generation between offload and non-offload modes for a failing sample:

```python
# Compare logits at each decode step
# Check if divergence starts at a specific position
# Log KV cache contents at divergence point
```

### Step 3: Verify KV Cache Contents

Add debugging to `OffloadEngine`:

```python
# In store_kv(): Log what's being stored
# In load_kv(): Log what's being loaded
# Compare loaded KV with expected values
```

### Step 4: Check Position/Index Calculations

```python
# Log ring buffer write/read positions
# Log CPU block indices
# Verify position IDs match actual token positions
```

### Step 5: Isolate the Bug

1. Test with shorter sequences (16K, 8K) to see if issue is length-dependent
2. Test with single chunk (no chunking) to isolate chunked prefill
3. Test prefill-only (no decode) to isolate decode phase

## Quick Debugging Commands

```bash
# Test single failing sample with verbose output
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --sample-indices 1 \
    --verbose

# Test with different context lengths
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH python tests/test_ruler_niah.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --enable-offload \
    --max-model-len 16384 \
    --verbose
```

## Related Documentation

- [`docs/ruler_niah_standalone_test.md`](ruler_niah_standalone_test.md) - Test setup and background
- [`docs/layerwise_offload_memory_analysis.md`](layerwise_offload_memory_analysis.md) - Memory analysis (if exists)

## Test Results Log

**Date**: 2025-01-12

| Test | Mode | Samples | Passed | Accuracy |
|------|------|---------|--------|----------|
| RULER NIAH 32K | Non-Offload | 100 | 100 | 100% |
| RULER NIAH 32K | CPU Offload | 100 | 66 | 66% |

## Next Steps

1. [ ] Identify pattern in failing samples (position of needle? specific numbers?)
2. [ ] Add detailed logging to offload engine
3. [ ] Compare logits between offload and non-offload modes
4. [ ] Bisect the code to find the exact bug location
5. [ ] Write unit test that isolates the bug
