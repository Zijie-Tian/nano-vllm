---
description: Mandatory GPU selection, allocation, and hardware rules
alwaysApply: true
---
# GPU Selection Protocol (MANDATORY)
1. Check Context: Look for explicit GPU IDs in the user's prompt or history.
2. Ask if Unclear: If no GPU is specified, **STOP and ASK** the user which GPU(s) to use.
3. Prefix Commands: Always use `CUDA_VISIBLE_DEVICES=X` for most GPU-related commands.
4. Script Exception: For `scripts/profile_offload.sh`, use the `--gpu X` argument instead of `CUDA_VISIBLE_DEVICES`.

# Multi-GPU Debugging & Resource Allocation
- Conservative Allocation: For long-running validations (>20 mins or >50 samples), use only 1-2 GPUs.
- Parallel Exploration: Reserve remaining GPUs (at least 50% if ≥4 GPUs are available) for parallel hypothesis testing and fast iteration (≤10 samples).
- Single-Task Validation: During debugging, focus on ONE representative task first. Do not run the full benchmark suite until the fix is verified.
- Non-Blocking Monitoring: Always run long tests in the background (`is_background: true`) and continue analysis or exploratory tests on other GPUs.

# GPU Testing & Hardware Rules
- RTX 3090 / 4090 (24GB): MUST use `--enable-offload`. GPU-only mode will OOM for 7B+ models.
- A100 (40/80GB): Both offload and GPU-only modes are acceptable.
- Needle Test Requirement: Always use `--enable-offload --input-len 32768`. 8K is insufficient for validation.
- GPU Mutex: Before running benchmarks (`bench*.py`), wait for exclusive GPU access if other processes are using the compute apps. Ensure exclusive GPU access by checking for other compute-intensive PIDs before running benchmarks.

# Monitoring
- GPU Monitoring: For profiling or OOM debugging, run monitoring commands in the background. Prefer `nvidia-smi` queries or specialized profiling tools (nsys) directed to background output files.
