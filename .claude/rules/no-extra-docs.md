# Documentation Policy

## Do Not Create Unnecessary Documentation

**IMPORTANT**: Do NOT create extra markdown documentation files unless explicitly requested by the user.

### What NOT to do:

- ❌ Do NOT create README files proactively
- ❌ Do NOT create analysis documents (*.md) after completing tasks
- ❌ Do NOT create tutorial/guide documents
- ❌ Do NOT create summary documents

### What TO do:

- ✅ Only create documentation when user explicitly asks for it
- ✅ Provide information directly in conversation instead
- ✅ Update existing documentation if changes require it
- ✅ Add inline code comments where necessary

### Exceptions:

Documentation is acceptable ONLY when:
1. User explicitly requests "create a README" or "write documentation"
2. Updating existing documentation to reflect code changes
3. Adding inline comments/docstrings to code itself

### Examples:

**Bad** (Don't do this):
```
User: "Profile the code"
Assistant: [Creates profiling_results.md after profiling]
```

**Good** (Do this instead):
```
User: "Profile the code"
Assistant: [Runs profiling, shows results in conversation]
```
