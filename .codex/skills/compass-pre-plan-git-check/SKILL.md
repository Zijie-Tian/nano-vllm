---
name: compass-pre-plan-git-check
description: Pre-implementation remote sync check for COMPASS. Use before feature work, bug fixes, or refactors. Source of truth: `.codex/rules/pre-plan-git-check.md`.
---

# COMPASS Pre-Plan Git Check

This skill keeps Codex aligned with `.codex/rules/pre-plan-git-check.md`.

## Required first step
Read `.codex/rules/pre-plan-git-check.md`.

## Enforce
- Run `git fetch origin` before starting implementation-focused work.
- Check whether the current branch is behind or diverged from the remote branch.
- If the branch is behind or diverged, stop and ask the user before pulling/rebasing or continuing.
- Do not auto-merge or auto-rebase on the user's behalf.

