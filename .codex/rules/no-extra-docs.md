# Documentation Policy

## Do Not Create Unnecessary Documentation

**IMPORTANT**: Do NOT create extra markdown documentation files proactively unless:
1. User explicitly requests documentation
2. Refactoring `AGENTS.md` to move Codex-visible technical details to `docs/` (see `doc-management.md`)
3. A shared cross-tool doc workflow requires index synchronization after adding a real technical document

### What NOT to do:

- Do NOT create README files proactively
- Do NOT create standalone analysis documents after completing tasks
- Do NOT create summary documents without request

### What TO do:

- Provide information directly in conversation by default
- When user requests documentation, follow `doc-management.md` workflow
- Update existing docs in `docs/` when code changes affect them
- Keep `AGENTS.md` concise, move technical details to `docs/`
- If a new doc is created, sync the doc index in `AGENTS.md` (and other tool entrypoints that maintain the shared docs table)

### Documentation Locations:

| Type | Location |
|------|----------|
| Operational requirements (Codex) | `AGENTS.md` |
| Technical details | `docs/*.md` |
| Code comments | Inline in source |

### Examples:

**Proactive docs (Don't do)**:
```
User: "Profile the code"
Assistant: [Creates profiling_results.md without being asked]
```

**On-request docs (Do this)**:
```
User: "Profile the code and document the findings"
Assistant: [Runs profiling, creates/updates docs/profiling_guide.md]
```

**Refactoring (Do this)**:
```
User: "AGENTS.md is too long, refactor it"
Assistant: [Moves technical sections to docs/, updates AGENTS.md index]
```
