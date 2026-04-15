---
name: compass-ruler-results-summarizer
description: Structured Codex wrapper for the existing repo-local RULER results summarizer. Use when aggregating multiple `summary.csv` files. Canonical implementation: `.agents/skills/ruler_results_summarizer/SKILL.md`.
---

# COMPASS RULER Results Summarizer

This skill exposes the existing repo-local summarizer through `.codex/` so the Codex environment stays structured.

## Required first step
Read `.agents/skills/ruler_results_summarizer/SKILL.md`.

## Enforce
- Prefer the canonical implementation and scripts that already live under `.agents/skills/ruler_results_summarizer/`.
- Reuse `scripts/aggregate_ruler.py` instead of rewriting the aggregation logic by hand.
- Treat `.agents/skills/ruler_results_summarizer/` as the implementation source of truth.

