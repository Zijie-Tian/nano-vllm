# Testing

## Test File Guidelines

### Naming Convention

- All test files should be named `test_*.py`
- Example: `test_attention.py`, `test_compass.py`

### Purpose

Tests are **educational scripts** for understanding module behavior:
- Focus on demonstrating how modules work
- Show the flow and interaction between components
- Help developers understand implementation details

### Code Style

1. **Script-based structure**: Write tests as executable scripts
2. **Utility functions**: Extract reusable steps as helper functions at the top of the file
3. **Main flow as script**: The actual test/demonstration logic runs as top-level script code

```python
# Example structure:

import torch
from compass.src.Compass import Compass

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
model = Compass(param=value)

# 2. Test feature X
result = model.forward(input)
assert result.shape == expected_shape

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

Use PYTHONPATH for isolation:

```bash
# Run a specific test
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH python tests/test_compass.py

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH python tests/test_attention.py
```

## RULER Benchmark

```bash
# Activate environment
conda activate ruler

# Run full benchmark
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full

# Run single task
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1
```

## Quick Verification

```bash
# Import test
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH python -c "from compass.src.Compass import Compass; print('OK')"

# Check flash_attn
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"

# Check flashinfer
python -c "import flashinfer; print('flashinfer: OK')"
```
