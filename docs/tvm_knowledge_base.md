# TVM Knowledge Base

This document records common issues, solutions, and optimization techniques for TVM-based operators in Nano-vLLM.

## 1. Fallback Configuration Warnings

### Issue
Warning: `WARNING: Cannot find config for target=..., workload=... A fallback configuration is used, which may bring great performance regression.`

### Cause
TVM's AutoTVM subsystem cannot find a pre-tuned optimization log for the specific operator shape and hardware target. It falls back to a generic, non-optimized schedule.

### Impact
Significant performance degradation (2x to 10x slower). It does not affect correctness but fails to utilize hardware features like SIMD vectorization and optimal cache layouts.

### Solution
1.  **Shared Logs**: Ensure template names do not include dimensions that don't affect the inner loop schedule (like sequence length `M`). This allows logs to be reused across different sequence lengths.
2.  **Explicit Loading**: In `OpCodegen.compile()`, explicitly load the best configuration from `tune.log` using fuzzy matching on the template name to bypass strict workload signature matching.
3.  **Automatic Tuning**: Run the benchmark script with `--tune` once to generate the optimized logs.

## 2. x86 Stability and Segfaults

### Issue
Segmentation faults occurring during execution of large sequences (e.g., 32k+ tokens) on x86_64 systems.

### Cause
A data type mismatch between the Python codegen and the underlying C++ intrinsics. Specifically, `tbl.cc` uses 32-bit `float` for x86 SIMD operations, but the Python side may default to `float16`.

### Solution
1.  **Architecture Detection**: Detect the target architecture in the operator's constructor.
2.  **Type Promotion**: Automatically switch `out_dtype` from `float16` to `float32` when running on x86_64.
3.  **Compiler Flags**: Ensure essential compiler flags like `-mavx2` and `-mfma` are passed to `cc_opts`.

## 3. Template Registration Errors

### Issue
`ValueError: Customized func is already registered in autoTVM task ...`

### Cause
Attempting to register the same `@autotvm.template` multiple times within the same Python process, which often happens when calling `compile` repeatedly with different shapes.

### Solution
Use a global dictionary cache (e.g., `_GLOBAL_TEMPLATE_CACHE`) in `base.py` to ensure each template name is only registered once per process.

---
*Last Updated: 2026-03-09*
