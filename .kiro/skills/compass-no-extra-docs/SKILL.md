---
name: compass-no-extra-docs
description: "Guardrail against unnecessary markdown creation in COMPASS. Use when deciding whether to create docs, summaries, or analysis writeups. Source of truth: `.codex/rules/no-extra-docs.md`."
---

# COMPASS No Extra Docs

This skill keeps Codex aligned with `.codex/rules/no-extra-docs.md`.

## Required first step
Read `.codex/rules/no-extra-docs.md`.

## Enforce
- Do not proactively create README or standalone markdown summary files.
- Prefer answering directly in the conversation unless the user explicitly asks for documentation or an existing workflow requires doc synchronization.
- When documentation is necessary, update the appropriate existing location instead of inventing a new one.
