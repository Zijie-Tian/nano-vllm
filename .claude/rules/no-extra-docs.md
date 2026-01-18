# Documentation Policy

## Do Not Create Unnecessary Documentation

**IMPORTANT**: Do NOT create extra markdown documentation files proactively unless:
1. User explicitly requests documentation
2. Refactoring CLAUDE.md to move technical details to docs/ (see `doc-management.md`)

### What NOT to do:

- Do NOT create README files proactively
- Do NOT create standalone analysis documents after completing tasks
- Do NOT create summary documents without request

### What TO do:

- Provide information directly in conversation by default
- When user requests documentation, follow `doc-management.md` workflow
- Update existing docs in `docs/` when code changes affect them
- Keep CLAUDE.md concise (< 150 lines), move technical details to docs/

### Documentation Locations:

| Type | Location |
|------|----------|
| Operational requirements | CLAUDE.md |
| Technical details | docs/*.md |
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
Assistant: [Runs profiling, creates/updates docs/memory_analysis.md]
```

**Refactoring (Do this)**:
```
User: "CLAUDE.md is too long, refactor it"
Assistant: [Moves technical sections to docs/, updates CLAUDE.md index]
```
