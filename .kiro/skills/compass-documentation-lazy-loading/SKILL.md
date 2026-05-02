---
name: compass-documentation-lazy-loading
description: "Keep COMPASS agent-entry docs lightweight through lazy-loaded references. Use when editing AGENTS/CLAUDE/doc indexes or deciding where detailed instructions should live. Source of truth: `.codex/rules/documentation-lazy-loading.md`."
---

# COMPASS Documentation Lazy Loading

This skill keeps Codex aligned with `.codex/rules/documentation-lazy-loading.md`.

## Required first step
Read `.codex/rules/documentation-lazy-loading.md`.

## Enforce
- Keep entry documents concise.
- Move long or scenario-specific detail into `docs/`.
- Maintain a clear reference table so detailed docs are loaded only when needed.
- Prefer links/index entries over copying long technical sections into entrypoint files.
