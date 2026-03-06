# GEMINI.md (Tests)

## Test Code Style Mandates

1.  **Minimalist Output**: Only the final `test_xxx: PASSED` should be printed on success. No intermediate debug prints unless specifically debugging.
2.  **Structured Inputs**: Use predictable data (all 1s, patterns) for easy manual verification.
3.  **Documentation of Logic**: Use comments to explain the expected value calculation *before* the `assert`.
4.  **Verification via `assert`**: Always use `assert actual == expected, f"Failed: {actual} != {expected}"`.

## Running Tests

- Always prefix with `CUDA_VISIBLE_DEVICES=X` (unless running CPU-only).
- Always use `PYTHONPATH=$(pwd):$PYTHONPATH`.

## test_ruler.py Quick Reference

- **Mandatory Read**: `docs/test_ruler_usage_guide.md`.
- **No `--help`**: Refer to docs instead.
- **max-model-len mapping**:
    - `ruler_32k`: 40960
    - `ruler_64k`: 72000
    - `ruler_128k`: 135000
- **3090/4090**: Must use `--enable-offload`.
