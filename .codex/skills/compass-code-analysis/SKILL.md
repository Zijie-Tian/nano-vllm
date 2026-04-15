---
name: compass-code-analysis
description: COMPASS code-navigation and call-chain analysis conventions. Use when tracing behavior, locating symbols, or preparing refactors. Source of truth: `.codex/rules/code-analysis.md`.
---

# COMPASS Code Analysis

This skill keeps Codex aligned with `.codex/rules/code-analysis.md`.

## Required first step
Read `.codex/rules/code-analysis.md`.

## Preferred tool order
1. Serena MCP when available.
2. Other symbol-aware Codex/code-intel/MCP tools.
3. Targeted text search only as a fallback.

## Expectations
- Prefer symbol-level navigation over broad grep/glob scans.
- Trace real call chains and cite concrete files/symbols.
- Before editing, make sure the collected evidence is sufficient for the requested change.

