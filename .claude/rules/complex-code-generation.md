# Complex Code Generation Rule

## Purpose

This rule governs the use of Codex MCP for complex code generation tasks including Triton kernel design, CUDA programming, and SIMD optimization. The goal is to leverage Codex's specialized capabilities for low-level performance code while maintaining quality through iterative testing and feedback.

---

## When to Apply This Rule

**MUST** use Codex MCP when implementing:

| Task Type | Examples |
|-----------|----------|
| Triton Kernels | Custom attention kernels, fused operations, block-sparse kernels |
| CUDA Programming | Custom CUDA kernels, kernel fusion, memory optimization |
| SIMD Optimization | AVX/AVX2/AVX-512 vectorized code, NEON intrinsics |
| Low-level Optimizations | Block-wise algorithms, warp-level primitives, shared memory tuning |
| Performance-critical Code | Kernel auto-tuning, pipeline optimization, memory coalescing |

---

## Mandatory Requirements

### 1. Model Specification (CRITICAL)

**MUST** always use `gpt-5.4` model:

```python
# Correct: Explicit model specification
mcp__codex-cli__codex(
    prompt="...",
    model="gpt-5.4",
    ...
)
```

### 2. Codex MCP Availability Check

Before starting complex code tasks, verify Codex MCP is available:

```python
# Check connectivity
mcp__codex-cli__ping(message="ping")
```

If Codex MCP is unavailable:
1. Fall back to manual implementation
2. Document the fallback in findings
3. Consider retrying Codex later for comparison

---

## Workflow

### Phase 1: Requirements Specification

**My Responsibility**: Provide comprehensive design specifications to Codex

Required specification components:

```markdown
## Design Specification Template

### 1. Algorithm Overview
- What: High-level algorithm description
- Why: Performance motivation and expected gains
- Where: Integration point in existing codebase

### 2. Interface Definition
```python
# Expected function signature
def kernel_name(input1: torch.Tensor, input2: torch.Tensor, ...) -> torch.Tensor:
    """
    Args:
        input1: [shape] dtype, description
        input2: [shape] dtype, description
    Returns:
        output: [shape] dtype, description
    """
```

### 3. Constraints & Requirements
| Category | Requirement |
|----------|-------------|
| Block size | Must be power of 2, preferably 128-512 |
| Memory | Max shared memory per block: XX KB |
| Precision | FP16/BF16/FP32 requirements |
| Numerical | Tolerance for numerical errors |

### 4. Algorithm Steps
1. Step 1: Load data from global memory
2. Step 2: Compute intermediate results
3. Step 3: Store results back

### 5. Reference Implementation (if available)
```python
# Naive PyTorch reference for validation
def reference_impl(...):
    ...
```

### 6. Performance Targets
- Minimum speedup vs baseline: X%
- Memory bandwidth utilization: Y%
- Maximum latency: Z us
```

### Phase 2: Code Generation via Codex

**Codex's Responsibility**: Generate optimized code based on specifications

```python
mcp__codex-cli__codex(
    prompt="""
    [Insert complete Design Specification from Phase 1]

    Please implement a Triton kernel that:
    1. Follows the algorithm steps exactly
    2. Optimizes for the specified block size
    3. Handles edge cases (partial blocks, boundary conditions)
    4. Includes detailed comments explaining the optimization strategy
    5. Provides both the kernel and a Python wrapper function

    Output format:
    - Complete, runnable Python code
    - Type hints where appropriate
    - Documentation strings
    - Example usage
    """,
    model="gpt-5.4",
    reasoningEffort="high",  # Use high effort for complex kernels
    sandbox="workspace-write"
)
```

### Phase 3: Testing & Validation

**My Responsibility**: Test the generated code thoroughly

Mandatory test categories:

| Test Type | Purpose | Criteria |
|-----------|---------|----------|
| Correctness | Verify output matches reference | Max error < tolerance |
| Shape Coverage | Test various input dimensions | Edge cases (1, small, large, non-power-of-2) |
| Numerical Stability | Check for NaN/Inf | No invalid values |
| Performance | Measure speedup | Meet or exceed targets |
| Memory Safety | Check for OOB access | No CUDA errors |

Test script template:

```python
"""
Test: [Kernel Name]

Validation suite for Codex-generated Triton kernel.
"""
import torch
import triton
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

from nanovllm.ops.[kernel] import [kernel_func]

# ============================================================
# Parameters
# ============================================================
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_SHAPES = [
    (128, 64),      # Small
    (1024, 1024),   # Medium
    (8192, 8192),   # Large
    (1000, 500),    # Non-power-of-2
]
ATOL = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}

# ============================================================
# Reference Implementation
# ============================================================
def reference_fn(x, y):
    """PyTorch reference for validation."""
    return ...

# ============================================================
# Correctness Tests
# ============================================================
def test_correctness():
    for dtype in TEST_DTYPES:
        for shape in TEST_SHAPES:
            x = torch.randn(shape, dtype=dtype, device="cuda")
            y = torch.randn(shape, dtype=dtype, device="cuda")

            expected = reference_fn(x, y)
            actual = kernel_func(x, y)

            atol = ATOL[dtype]
            assert torch.allclose(actual, expected, atol=atol), \
                f"Failed: dtype={dtype}, shape={shape}"

    print(f"test_correctness: PASSED")

# ============================================================
# Performance Benchmark
# ============================================================
def benchmark():
    # Warmup
    for _ in range(10):
        ...

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        ...
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / 100
    print(f"Benchmark: {elapsed_ms:.3f} ms/iter")

if __name__ == "__main__":
    test_correctness()
    benchmark()
```

### Phase 4: Feedback Loop

**My Responsibility**: Provide detailed feedback to Codex for refinement

If tests fail, use this feedback template:

```markdown
## Test Feedback for Refinement

### Issue Summary
- Test: [Which test failed]
- Input: [Shape, dtype, configuration]
- Error: [Error message or numerical deviation]

### Expected Behavior
[What the correct output should be]

### Actual Behavior
[What the current code produces]

### Root Cause Analysis (if known)
[Your analysis of why it fails]

### Suggested Fix (if known)
[Specific suggestions for Codex]

### Current Code
```python
[paste relevant code section]
```
```

Then request refinement:

```python
mcp__codex-cli__codex(
    prompt="""
    The previous kernel has issues. Please fix:

    [Insert Feedback from above]

    Please provide the corrected implementation.
    """,
    model="gpt-5.4",
    reasoningEffort="high",
    sandbox="workspace-write"
)
```

---

## Code Quality Standards

### Generated Code Must Include

| Element | Requirement |
|---------|-------------|
| Docstrings | Full description of inputs, outputs, behavior |
| Comments | Explain non-obvious optimizations |
| Type Hints | For all public functions |
| Error Handling | Validate inputs, raise meaningful errors |
| Edge Cases | Handle empty tensors, boundary conditions |
| Numerical Stability | Use appropriate accumulation types |

### Triton-specific Requirements

```python
# Good Triton kernel characteristics:

@triton.jit
def good_kernel(
    input_ptr, output_ptr,
    stride_m, stride_n,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Clear description of what this kernel does.

    Args:
        input_ptr: Pointer to input tensor [M, N]
        output_ptr: Pointer to output tensor [M, N]
        ...
    """
    # 1. Program ID calculation with clear variable names
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Block-level offset calculation
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 3. Mask creation for boundary handling
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 4. Memory coalesced loads
    input_block = tl.load(input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask)

    # 5. Computation
    output_block = ...

    # 6. Guarded stores
    tl.store(output_ptr + ..., output_block, mask=mask)
```

---

## Integration with Existing Codebase

### File Organization

Place generated kernels in appropriate directories:

```
nanovllm/ops/
├── triton_kernels/     # Pure Triton kernels
│   ├── __init__.py
│   ├── attention.py
│   ├── softmax.py
│   └── [new_kernel].py
├── cuda/               # Custom CUDA kernels
│   ├── __init__.py
│   └── [kernel].cu
└── simd/               # SIMD optimized code
    ├── __init__.py
    └── [impl].py
```

### Registration Pattern

```python
# In nanovllm/ops/__init__.py

def get_kernel_impl(name: str):
    """Get optimal kernel implementation for current hardware."""
    if torch.cuda.is_available() and name in TRITON_KERNELS:
        return TRITON_KERNELS[name]
    return FALLBACK_KERNELS[name]
```

---

## Example: Complete Workflow

### Task: Implement Block-Sparse Attention Kernel

**Step 1**: Prepare specification
```markdown
## Design Spec: Block-Sparse Attention Kernel

### Algorithm
Compute attention only for selected block pairs...

### Interface
```python
def block_sparse_attention(
    q: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    k: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    v: torch.Tensor,          # [batch, num_heads, seq_len, head_dim]
    block_indices: torch.Tensor,  # [batch, num_heads, num_blocks, 2]
    block_size: int = 128,
) -> torch.Tensor:            # [batch, num_heads, seq_len, head_dim]
```

### Constraints
- BLOCK_SIZE must be multiple of 64
- head_dim must be 64 or 128
- seq_len must be divisible by BLOCK_SIZE
```

**Step 2**: Call Codex MCP
```python
mcp__codex-cli__codex(
    prompt="[specification from Step 1]",
    model="gpt-5.4",
    reasoningEffort="high",
    sandbox="workspace-write"
)
```

**Step 3**: Create test file
```python
# tests/test_block_sparse_attention.py
# [Test implementation following template]
```

**Step 4**: Run tests
```bash
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_block_sparse_attention.py
```

**Step 5**: If failures, provide feedback and iterate

---

## Performance Validation Checklist

Before accepting generated code:

- [ ] Correctness: Matches reference within tolerance
- [ ] Numerical: No NaN/Inf for any valid input
- [ ] Memory: No OOB accesses detected by compute-sanitizer
- [ ] Performance: Meets speedup targets
- [ ] Integration: Works with existing pipeline
- [ ] Documentation: Fully documented API
- [ ] Edge cases: Handles all boundary conditions

---

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Vague requirements | Provide specific shapes, dtypes, constraints |
| Skip reference implementation | Always provide PyTorch reference for validation |
| Accept without testing | Run full test suite before integration |
| Ignore numerical errors | Investigate and fix precision issues |
| Skip edge case handling | Test and handle boundary conditions |
| Use default model | Explicitly specify `gpt-5.4` |

---

## Summary

| Phase | Responsible | Action |
|-------|-------------|--------|
| 1. Spec | Me | Write detailed design specification |
| 2. Generate | Codex (gpt-5.4) | Generate optimized kernel code |
| 3. Test | Me | Run correctness and performance tests |
| 4. Feedback | Me → Codex | Iterate until all tests pass |
| 5. Integrate | Me | Merge into codebase with proper structure |

---

**Author**: Zijie Tian
**Version**: 1.0
