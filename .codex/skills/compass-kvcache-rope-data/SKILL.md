---
name: compass-kvcache-rope-data
description: "KVCache/RoPE data access workflow for COMPASS. Use when code or analysis depends on files under `results/kvcache-rope/`. Source of truth: `.codex/rules/kvcache-rope-data.md`."
---

# COMPASS KVCache/RoPE Data

This skill keeps Codex aligned with `.codex/rules/kvcache-rope-data.md`.

## Required first step
Read `.codex/rules/kvcache-rope-data.md`.

## Enforce
- Check the exact local file path first.
- If the file is missing, use the `aliyunpan` skill instead of inventing a different download workflow.
- After download, verify the final local file exists and roughly matches the expected size before running dependent code.
- Only download the layers/files that the current task actually needs.
