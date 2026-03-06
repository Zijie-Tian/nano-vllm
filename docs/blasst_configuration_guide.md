# BLASST Configuration Guide

## Overview

This document describes the configuration parameters for BLASST (Dynamic BLocked Attention Sparsity via Softmax Thresholding) sparse attention policy.

**Status**: ✅ Completed
**Related Files**:
- `nanovllm/config.py` - Configuration parameters
- `nanovllm/kvcache/sparse/blasst.py` - Policy implementation
- `nanovllm/kvcache/__init__.py` - Factory function

---

## Configuration Parameters

BLASST parameters are defined in `nanovllm/config.py` under the `Config` dataclass:

```python
# BLASST specific parameters
blasst_a: int = 16384                  # Inverse formula numerator (λ = a / L)
blasst_fixed_lambda: float | None = 0.5  # Fixed threshold instead of formula
blasst_granularity: int = 128          # Token granularity for skip decisions
```

---

## Parameter Details

### `blasst_a`

**Type**: `int`
**Default**: `16384`

The numerator in the inverse formula for computing the dynamic threshold λ:

```
λ = a / L
```

Where:
- `a` is the configurable numerator (default 16384)
- `L` is the current sequence length

**Examples**:
| Sequence Length | λ (with a=16384) |
|-----------------|------------------|
| 32K (32768)     | 0.5              |
| 64K (65536)     | 0.25             |
| 128K (131072)   | 0.125            |

**When to adjust**:
- Increase `a` for more aggressive skipping (higher λ, more blocks skipped)
- Decrease `a` for more conservative skipping (lower λ, fewer blocks skipped)

---

### `blasst_fixed_lambda`

**Type**: `float | None`
**Default**: `0.5`

When set, uses a fixed threshold value instead of the dynamic formula. When `None`, falls back to the formula `λ = a / L`.

**Behavior**:
| Value | Effect |
|-------|--------|
| `0.5` (default) | Skip blocks where `local_max - running_max < ln(0.5) ≈ -0.693` |
| `0.3` | More aggressive skipping |
| `0.7` | More conservative skipping |
| `None` | Use dynamic formula `λ = a / L` |

**Recommendation**:
- Use fixed values (e.g., 0.5) for consistent behavior across different sequence lengths
- Use `None` (formula mode) when you want automatic adaptation to sequence length

---

### `blasst_granularity`

**Type**: `int`
**Default**: `128`

Token granularity for BLASST skip decisions. Both query and KV are processed at this granularity.

**Common Values**:
| Granularity | Description | Use Case |
|-------------|-------------|----------|
| `64` | Fine-grained | Maximum sparsity precision |
| `128` (default) | Standard | Balance precision and performance |
| `256` | Coarse | Lower overhead, less precise |
| `1024` | Block-level | Minimal overhead, original design |

**Impact**:
- Smaller granularity → More precise skip decisions → Higher sparsity
- Smaller granularity → More kernel launches → Higher latency
- Larger granularity → Less precise skip decisions → Lower sparsity
- Larger granularity → Fewer kernel launches → Lower latency

---

## Usage Examples

### Python API

```python
from nanovllm.config import Config, SparsePolicyType

# Default configuration (fixed_lambda=0.5, granularity=128)
config = Config(
    model="~/models/Llama-3.1-8B-Instruct",
    sparse_policy=SparsePolicyType.BLASST,
)

# Custom configuration - more aggressive skipping
config = Config(
    model="~/models/Llama-3.1-8B-Instruct",
    sparse_policy=SparsePolicyType.BLASST,
    blasst_fixed_lambda=0.3,      # More aggressive skipping
    blasst_granularity=128,
)

# Use dynamic formula instead of fixed value
config = Config(
    model="~/models/Llama-3.1-8B-Instruct",
    sparse_policy=SparsePolicyType.BLASST,
    blasst_fixed_lambda=None,     # Use λ = a / L formula
    blasst_a=8192,                # Adjust formula parameter
    blasst_granularity=128,
)

# Fine-grained precision
config = Config(
    model="~/models/Llama-3.1-8B-Instruct",
    sparse_policy=SparsePolicyType.BLASST,
    blasst_fixed_lambda=0.5,
    blasst_granularity=64,        # Finer granularity
)
```

### Command Line (via test_ruler.py)

Currently, parameters must be set in code. For command-line usage, modify the defaults in `config.py` or use a custom script.

```bash
# Standard usage (uses defaults from config)
python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --enable-offload \
    --sparse-policy BLASST
```

---

## Algorithm Details

### Skip Condition

BLASST determines whether to skip a KV block using:

```python
skip = (local_max - running_max) < ln(λ)
```

Where:
- `local_max`: Maximum attention score for current KV block
- `running_max`: Running maximum across all processed blocks so far
- `λ`: Threshold (either fixed or computed via formula)

### Two Threshold Modes

#### 1. Fixed Lambda Mode (default)

```python
λ = blasst_fixed_lambda  # e.g., 0.5
```

**Advantages**:
- Consistent behavior across different sequence lengths
- Easier to tune for specific accuracy requirements

#### 2. Dynamic Formula Mode

```python
λ = blasst_a / seq_len   # e.g., 16384 / 32768 = 0.5
```

**Advantages**:
- Automatically adapts to sequence length
- Longer sequences use smaller thresholds (more aggressive skipping)

---

## Tuning Guide

### For Maximum Accuracy

```python
blasst_fixed_lambda=0.7   # Conservative skipping
blasst_granularity=64     # Fine-grained decisions
```

**Expected**: ~95% skip rate, highest accuracy

### For Maximum Speed

```python
blasst_fixed_lambda=0.3   # Aggressive skipping
blasst_granularity=256    # Coarse decisions
```

**Expected**: ~99% skip rate, fastest inference

### Balanced (Recommended)

```python
blasst_fixed_lambda=0.5   # Default
blasst_granularity=128    # Default
```

**Expected**: ~96-98% skip rate, good accuracy/speed balance

---

## Configuration Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `blasst_a` | `int` | `16384` | > 0 | Formula numerator for dynamic λ |
| `blasst_fixed_lambda` | `float \| None` | `0.5` | 0.0-1.0 or `None` | Fixed threshold (overrides formula) |
| `blasst_granularity` | `int` | `128` | 64, 128, 256, 512, 1024 | Token granularity for skip decisions |

---

## Verification

Test your configuration:

```python
from nanovllm.config import Config, SparsePolicyType
from nanovllm.kvcache import create_kvcache_manager

config = Config(
    model="~/models/Llama-3.1-8B-Instruct",
    sparse_policy=SparsePolicyType.BLASST,
    blasst_fixed_lambda=0.4,
    blasst_granularity=128,
    enable_cpu_offload=False,
    max_model_len=4096,
)

manager = create_kvcache_manager(config)
print(manager.sparse_policy)
# Output: BLASSTPolicy(fixed_lambda=0.4, granularity=128)
```

---

## Related Documentation

- `docs/blasst_implementation_report.md` - BLASST implementation overview
- `docs/blasst_128_granularity.md` - Fine-grained granularity details
- `docs/sparse_attention_guide.md` - General sparse attention methods

---

**Summary**: BLASST provides three configurable parameters (`a`, `fixed_lambda`, `granularity`) for fine-tuning the sparsity-accuracy trade-off. The default values (`fixed_lambda=0.5`, `granularity=128`) provide a good balance for most use cases.
