---
name: compass-gpu-testing
description: "COMPASS GPU test and benchmark safety rules. Use before running any GPU command, benchmark, or profiling workflow. Source of truth: `.codex/rules/gpu-testing.md`."
---

# COMPASS GPU Testing

This skill keeps Codex aligned with `.codex/rules/gpu-testing.md`.

## Required first step
Read `.codex/rules/gpu-testing.md`.

## Enforce
- Before any GPU command, require an explicit GPU id from the user if one was not provided.
- Always prefix GPU commands with `CUDA_VISIBLE_DEVICES=...`.
- Detect the GPU type first with `nvidia-smi --query-gpu=name --format=csv,noheader | head -1`.
- Make sure the benchmark environment, PYTHONPATH, and GPU constraints from the rule are satisfied before running tests.
