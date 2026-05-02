---
name: compass-ruler-task-config
description: "COMPASS RULER task configuration sync rules. Use when editing task lists or enabling/disabling RULER tasks. Source of truth: `.codex/rules/ruler-task-config.md`."
---

# COMPASS RULER Task Configuration

This skill keeps Codex aligned with `.codex/rules/ruler-task-config.md`.

## Required first step
Read `.codex/rules/ruler-task-config.md`.

## Enforce
- Update both `eval/RULER/scripts/config_tasks.sh` and `eval/RULER/scripts/eval.sh` together.
- Keep the `synthetic` arrays aligned between the two files.
- Disable tasks with comments instead of deleting task lines.
- Preserve the complete task list for future re-enabling.
