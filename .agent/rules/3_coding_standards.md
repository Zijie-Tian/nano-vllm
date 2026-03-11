---
description: Code optimization, sparse policy, and testing standards
glob: "*.py"
alwaysApply: true
---
# Sparse Policy Implementation
* No `None` Policy: `sparse_policy` must never be `None`. Default to `FullAttentionPolicy` if unspecified.
* OffloadEngine Communication: All CPU-GPU data transfers in policies MUST go through `OffloadEngine` (e.g., `load_to_slot_layer`, `wait_slot_layer`) to ensure stream synchronization and pipeline optimization. Direct `.to("cuda")` or `.copy_()` is prohibited in compute methods.
* Interface Compliance: Policies must declare `supports_prefill`/`supports_decode` and implement `select_blocks()`, `compute_chunked_prefill()`, and `compute_chunked_decode()`.

# Low-Level Optimization (Triton/CUDA)
* Design First: Always create a design specification before implementing Triton kernels, including algorithm overview, interface, and numerical constraints.
* Reference Implementation: Always provide a naive PyTorch reference for correctness validation.
* Validation: Kernels must be tested for correctness (matching reference), numerical stability (no NaN/Inf), and performance (speedup vs baseline).

# Testing & Profiling Standards
* Test Style: Minimal prints, use structured data (e.g., all ones) for easy manual verification, use `assert` for validation, and only print `test_xxx: PASSED` at the end.
* Profiling (nsys): MUST use `scripts/profile_offload.sh`. Do NOT run `nsys` directly.
* test_ruler.py: Read `docs/test_ruler_usage_guide.md` before running. Do not use `--help`. Match `data-dir` with appropriate `max-model-len`.
