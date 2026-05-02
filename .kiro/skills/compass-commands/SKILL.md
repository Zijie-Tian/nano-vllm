---
name: compass-commands
description: "COMPASS runtime commands, environment setup, and benchmark invocation rules. Use when activating conda, setting PYTHONPATH, running scripts, or checking imports."
---

# COMPASS Commands

## Required first step
Read `.codex/rules/commands.md` for complete command reference.

## Key Rules
- Use `PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH` instead of `pip install -e .`
- Activate `conda activate ruler` for benchmark workflows
- Prefer documented entrypoints: `./scripts/run_ruler.sh` or `eval/RULER/scripts/run.sh`
- Always require explicit GPU id from user before GPU commands
- Use documented import check to verify environment setup

## Source
`.codex/rules/commands.md`
