# Documentation Management

## AGENTS.md Content Policy

**AGENTS.md should only contain operational requirements:**
- Environment setup (PYTHONPATH, GPU mutex)
- Execution requirements (how to run tests/benchmarks)
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
├── architecture_guide.md      # Core components, design, flows
├── sparse_attention_guide.md  # Sparse attention methods
├── layerwise_offload_memory_analysis.md  # Memory analysis
├── debugging_guide.md         # Debugging techniques
└── <new_topic>_guide.md       # New technical topic
```

### Step 3: Update AGENTS.md Documentation Index

Add entry to the Documentation Index table:
```markdown
| Document | Purpose |
|----------|---------|
| [`docs/new_doc.md`](docs/new_doc.md) | Brief description |
```

### Step 4: Refactor if Needed

If AGENTS.md grows too large (> 150 lines), refactor:
1. Identify technical details that can be moved
2. Create appropriate doc in docs/
3. Replace detailed content with reference link
4. Keep only operational essentials in AGENTS.md

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
