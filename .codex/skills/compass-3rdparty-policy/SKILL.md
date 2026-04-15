---
name: compass-3rdparty-policy
description: COMPASS 3rdparty modification policy. Use before touching anything under `3rdparty/` or when considering dependency changes. Source of truth: `.codex/rules/3rdparty-policy.md`.
---

# COMPASS 3rdparty Policy

This skill keeps Codex aligned with `.codex/rules/3rdparty-policy.md`.

## Required first step
Read `.codex/rules/3rdparty-policy.md`.

## Enforce
- Treat `3rdparty/` as a fixed stable dependency area; do not modify vendored code in place.
- When a dependency needs changes, work in the dedicated upstream/development repository first.
- Only sync a validated dependency update back into COMPASS after testing.
- For `3rdparty/nanovllm`, keep the submodule on the `tzj/minference` branch when updating.

