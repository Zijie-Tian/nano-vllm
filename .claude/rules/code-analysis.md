# Code Analysis

## Use cclsp MCP for Code Navigation

When analyzing code, understanding call chains, or exploring the codebase, **prefer using the cclsp MCP tools** over grep/glob-based searches:

### Available cclsp Tools

| Tool | Purpose |
|------|---------|
| `mcp__cclsp__find_definition` | Jump to symbol definition |
| `mcp__cclsp__find_references` | Find all usages of a symbol |
| `mcp__cclsp__rename_symbol` | Rename a symbol across the codebase |
| `mcp__cclsp__get_diagnostics` | Get LSP diagnostics (errors, warnings) |
| `mcp__cclsp__restart_server` | Restart the LSP server if needed |

### When to Use cclsp

1. **Understanding call chains**: Use `find_references` to trace how functions are called
2. **Finding implementations**: Use `find_definition` to jump to actual code
3. **Refactoring**: Use `rename_symbol` for safe cross-file renames
4. **Code quality**: Use `get_diagnostics` to check for issues

### Example Workflow

```
1. User asks: "How does the prefill flow work?"
2. Use find_definition to locate key entry points (e.g., run_chunked_offload_prefill)
3. Use find_references to trace the call chain through the codebase
4. Read relevant code sections to understand the implementation
```

### Benefits over grep/glob

- **Semantic understanding**: cclsp understands code structure, not just text patterns
- **Accurate references**: Finds actual usages, not just text matches
- **Cross-file navigation**: Follows imports and definitions across modules
- **Type-aware**: Understands Python types and class hierarchies
