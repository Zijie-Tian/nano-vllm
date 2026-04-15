---
name: compass-testing
description: COMPASS testing style and verification conventions. Use when writing tests, choosing test filenames, or running verification scripts. Source of truth: `.codex/rules/testing.md`.
---

# COMPASS Testing

This skill keeps Codex aligned with `.codex/rules/testing.md`.

## Required first step
Read `.codex/rules/testing.md`.

## Enforce
- Use `test_*.py` naming.
- Prefer script-style educational tests with helper functions at the top and the main verification flow below.
- Minimize print output; prefer assertions and a final `PASSED` line when appropriate.
- Use the documented PYTHONPATH-based execution commands for verification.

