# Testing

## Test File Guidelines

### Naming Convention

- All test files must be named `test_*.py`
- Example: `test_offload_engine.py`, `test_ring_buffer.py`

### Purpose

Tests are **educational scripts** for understanding module behavior, NOT traditional unit tests:
- Focus on demonstrating how modules work
- Show the flow and interaction between components
- Help developers understand implementation details

### Code Style

1. **Script-based structure**: Write tests as executable scripts, not pytest-style functions
2. **Utility functions**: Extract reusable steps as helper functions at the top of the file
3. **Main flow as script**: The actual test/demonstration logic runs as top-level script code

```python
# Example structure:

import torch
from nanovllm.kvcache import SomeModule

# ============================================================
# Utility Functions
# ============================================================

def verify(tensor, expected, name):
    actual = tensor.mean().item()
    assert abs(actual - expected) < 0.01, f"{name}: {actual} != {expected}"

# ============================================================
# Main Test Script
# ============================================================

# 1. Initialize
module = SomeModule(param=value)

# 2. Test feature X
result = module.do_something()
assert result == expected_value

# 3. Test feature Y
...

print("test_xxx: PASSED")
```

### Comments

- Keep comments concise and clear
- Only add comments where the code isn't self-explanatory
- Use section headers (`# === Section ===`) to organize logical blocks

### Output

- **Minimize print statements** - the code should be self-explanatory
- Only print a final "PASSED" message at the end
- Use `assert` for verification instead of printing results
- If the user needs explanation, they will ask

## Running Tests

```bash
# Run a specific test
python tests/test_offload_engine.py

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python tests/test_ring_buffer.py
```

## Benchmarks

```bash
# Standard GPU benchmark
python bench.py

# CPU offload benchmark
python bench_offload.py

# vLLM comparison benchmark
python bench_vllm.py
```

## Quick Verification

```bash
# Import test
python -c "from nanovllm import LLM"

# Run offload benchmark (tests CPU-primary ring buffer mode)
python bench_offload.py
```
