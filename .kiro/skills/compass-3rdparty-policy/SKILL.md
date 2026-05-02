---
name: compass-3rdparty-policy
description: "3rdparty directory protection policy. Use before touching any files in 3rdparty/."
---

# COMPASS 3rdparty Policy

## Required first step
Read `.codex/rules/3rdparty-policy.md` for complete policy.

## Key Rules
- **NEVER** modify `3rdparty/` in place
- Update dependencies in their dedicated development repos first
- Sync stable copy/submodule back to COMPASS after testing
- Treat `3rdparty/` as read-only

## Source
`.codex/rules/3rdparty-policy.md`
