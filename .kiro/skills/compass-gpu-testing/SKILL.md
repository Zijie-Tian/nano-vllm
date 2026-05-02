---
name: compass-gpu-testing
description: "GPU testing and benchmark execution rules. Use for any GPU command, benchmark, or profiling task."
---

# COMPASS GPU Testing

## Required first step
Read `.codex/rules/gpu-testing.md` for complete GPU testing rules.

## Key Rules
- **ALWAYS** require explicit GPU id from user before any GPU command
- Prefix all GPU commands with `CUDA_VISIBLE_DEVICES=<gpu_id>`
- Never assume or auto-select GPU devices
- Use documented benchmark commands from rules

## Source
`.codex/rules/gpu-testing.md`
