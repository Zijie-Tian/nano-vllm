# Code Analysis

## Use Serena MCP for Code Navigation

When analyzing code, understanding call chains, or exploring the codebase, **prefer using the Serena MCP tools** over grep/glob-based searches:

### Available Serena Tools

| Tool | Purpose |
|------|---------|
| `mcp__serena__find_symbol` | Find symbol definitions by name path |
| `mcp__serena__find_referencing_symbols` | Find all usages of a symbol |
| `mcp__serena__get_symbols_overview` | Get high-level overview of symbols in a file |
| `mcp__serena__replace_symbol_body` | Replace symbol definition |
| `mcp__serena__insert_after_symbol` | Insert code after a symbol |
| `mcp__serena__insert_before_symbol` | Insert code before a symbol |
| `mcp__serena__rename_symbol` | Rename a symbol across the codebase |
| `mcp__serena__search_for_pattern` | Search for patterns in codebase |

### When to Use Serena

1. **Understanding call chains**: Use `find_referencing_symbols` to trace how functions are called
2. **Finding implementations**: Use `find_symbol` with `include_body=True` to get actual code
3. **Getting file overview**: Use `get_symbols_overview` to understand file structure first
4. **Refactoring**: Use `replace_symbol_body` for precise symbol-level edits
5. **Adding code**: Use `insert_after_symbol` or `insert_before_symbol` for clean insertions

### Example Workflow

```
1. User asks: "How does the attention computation work?"
2. Use get_symbols_overview to locate key files (e.g., compass/src/Compass.py)
3. Use find_symbol with depth=1 to find relevant classes and methods
4. Use find_referencing_symbols to trace the call chain
5. Read symbol bodies only when needed
```

### Benefits over grep/glob

- **Semantic understanding**: Serena understands code structure, not just text patterns
- **Accurate references**: Finds actual usages, not just text matches
- **Cross-file navigation**: Follows imports and definitions across modules
- **Type-aware**: Understands Python types and class hierarchies
- **Token efficient**: Read only what you need, not entire files

### Memory Integration

Use Serena's memory tools for persistent context:
- `mcp__serena__write_memory` - Store useful information
- `mcp__serena__read_memory` - Retrieve stored context
- `mcp__serena__list_memories` - See available memories

### Think Tools

Always use Serena's thinking tools for complex tasks:
- `mcp__serena__think_about_collected_information` - After gathering info
- `mcp__serena__think_about_task_adherence` - Before making changes
- `mcp__serena__think_about_whether_you_are_done` - When completing tasks
