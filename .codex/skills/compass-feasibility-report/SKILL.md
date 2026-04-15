---
name: compass-feasibility-report
description: "COMPASS feasibility-report workflow. Use for architecture trade-offs, proposal evaluation, or solution feasibility analysis. Source of truth: `.codex/rules/feasibility-report.md`; preferred agent: `.codex/agents/codex-deep-thinker.toml`."
---

# COMPASS Feasibility Report

This skill keeps Codex aligned with `.codex/rules/feasibility-report.md`.

## Required first step
Read both of these files before producing output:
- `.codex/rules/feasibility-report.md`
- `docs/FEASIBILITY_REPORT_TEMPLATE.md`

## Preferred execution surface
- Use the repo-local custom agent `.codex/agents/codex-deep-thinker.toml` for deep analysis work.

## Non-negotiable requirements
- Every report must include a feasibility score.
- Follow the required section structure from the template/rule.
- Include concrete integration points, quantified trade-offs, and explicit risks.
- Save the report under the requested `docs/` location only when the task actually calls for a written report artifact.
