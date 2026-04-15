---
name: compass-commands
description: "COMPASS runtime commands, environment setup, and benchmark invocation rules. Use when activating conda, setting PYTHONPATH, running scripts, or checking imports. Source of truth: `.codex/rules/commands.md`."
---

# COMPASS Commands

This skill keeps Codex aligned with `.codex/rules/commands.md`.

## Required first step
Read `.codex/rules/commands.md`.

## Enforce
- Use `PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH` instead of `pip install -e .`.
- Activate `conda activate ruler` for benchmark workflows that need the repo environment.
- Prefer the documented entrypoints such as `./scripts/run_ruler.sh ...` or `eval/RULER/scripts/run.sh`.
- Use the documented import check before claiming environment setup success.
