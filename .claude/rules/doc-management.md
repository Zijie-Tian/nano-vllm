# Documentation Management

## CLAUDE.md Content Policy

**CLAUDE.md should only contain operational requirements:**
- Environment setup (PYTHONPATH, conda)
- Execution requirements (how to run benchmarks)
- Quick configuration reference
- Documentation index (links to detailed docs)

**Technical details should go to docs/:**
- Architecture and design explanations
- Implementation details and code flows
- Debugging techniques
- Memory analysis and profiling
- Algorithm explanations

## When Adding New Technical Content

Follow this workflow:

### Step 1: Analyze and Document

If doing technical analysis (e.g., memory profiling):
1. Calculate theoretical values using formulas
2. Run actual tests to measure real values
3. Compare theoretical vs actual (expect < 10% error for valid models)
4. Document findings with both theory and empirical validation

### Step 2: Create/Update docs/

Create a new doc or update existing one in `docs/`:
```
docs/
├── CLAUDE_SETUP_TASK.md      # Environment setup guide
├── ENVIRONMENT_SETUP.md      # Detailed environment docs
├── <new_topic>_guide.md      # New technical topic
```

### Step 3: Update CLAUDE.md Documentation Index

Add entry to the Documentation References table:
```markdown
| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 新文档 | `docs/new_doc.md` | 简短描述 | 触发条件 |
```

### Step 4: Refactor if Needed

If CLAUDE.md grows too large (> 200 lines), refactor:
1. Identify technical details that can be moved
2. Create appropriate doc in docs/
3. Replace detailed content with reference link
4. Keep only operational essentials in CLAUDE.md

## Documentation Structure Template

For new technical docs:

```markdown
# Topic Guide

Brief overview of what this document covers.

## Section 1: Concepts
- Key concepts and terminology

## Section 2: Implementation
- Code locations
- Key methods/functions

## Section 3: Details
- Detailed explanations
- Code examples

## Section 4: Validation (if applicable)
- Theoretical analysis
- Empirical measurements
- Comparison table
```

## Memory Analysis Template

When documenting memory behavior:

```markdown
## Theoretical Calculation

| Component | Formula | Size |
|-----------|---------|------|
| Buffer X | `param1 × param2 × dtype_size` | X MB |

## Empirical Validation

| Metric | Theoretical | Actual | Error |
|--------|-------------|--------|-------|
| Peak memory | X GB | Y GB | Z% |

## Key Findings
1. Finding 1
2. Finding 2
```
