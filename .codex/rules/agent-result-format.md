# Agent Result Format Rules

## Purpose

Minimize token usage when background agents return results to the main agent. Raw program output is verbose and wastes context window space.

---

## 1. Result Formatting Principle

**MUST** return **structured summaries** instead of raw output.

| Don't | Do |
|-------|-----|
| Full program stdout/stderr | Key metrics only |
| Debug logs | Pass/Fail status |
| Verbose error stacks | Error summary + location |

---

## 2. Standard Result Templates

### 2.1 Test Results (RULER, Unit Tests, etc.)

```markdown
## Test Results: [Task Name]

**Pass Rate**: X / Y (Z%)

### Failed Samples (if any)
| Sample | Expected | Got |
|--------|----------|-----|
| N | expected_value | actual_value |

### Passed Samples
[List sample IDs or "All N samples passed"]
```

**Example** (instead of raw test output):
```markdown
## Test Results: niah_single_1 (Samples 0-49)

**Pass Rate**: 50 / 50 (100%)

### Passed Samples
All 50 samples passed.
```

### 2.2 Benchmark Results

```markdown
## Benchmark Results: [Task Name]

| Metric | Value |
|--------|-------|
| Throughput | X tok/s |
| Latency (p50) | Y ms |
| Latency (p99) | Z ms |
| Memory Peak | W GB |
```

### 2.3 Build/Compile Results

```markdown
## Build Results: [Target]

**Status**: SUCCESS / FAILED

### Errors (if any)
| File | Line | Error |
|------|------|-------|
| path/to/file.py | 123 | error message |
```

### 2.4 Investigation/Research Results

```markdown
## Investigation: [Topic]

### Findings
1. Finding 1 (with file:line reference)
2. Finding 2

### Relevant Files
- path/to/file1.py: description
- path/to/file2.py: description

### Conclusion
[1-2 sentence summary]
```

---

## 3. Mandatory Fields by Task Type

| Task Type | Required Fields |
|-----------|-----------------|
| Test Run | Pass/Fail count, failed sample details |
| Benchmark | Key metrics (throughput, latency, memory) |
| Build | Status, error locations |
| Search | File paths, line numbers, brief context |
| Verification | Before/After comparison, conclusion |

---

## 4. What to EXCLUDE

**MUST NOT** include in results:

| Exclude | Reason |
|---------|--------|
| Full stack traces | Extract error type + location only |
| Model loading logs | Not relevant to result |
| Progress bars / tqdm output | Noise |
| Warnings (unless critical) | Noise |
| Repeated successful outputs | "All X passed" is sufficient |
| Timestamps | Usually not needed |
| Device info (unless debugging hardware) | Noise |

---

## 5. Agent Prompt Template

When spawning background agents, include this instruction:

```
When reporting results, use a structured summary format:
- For tests: Pass rate, failed sample details (expected vs actual)
- For benchmarks: Key metrics table
- Do NOT include raw program output, logs, or verbose debug info
- Focus on actionable information only
```

---

## 6. Main Agent Instructions

When spawning a background agent for testing:

**Before** (verbose):
```
Run tests for samples 0-49 and report the output.
```

**After** (structured):
```
Run tests for samples 0-49. Report results as:
- Total pass/fail count
- For each failure: sample ID, expected value, actual value
- Do NOT include raw program output or logs
```

---

## 7. Examples

### Bad (Wastes ~500 tokens):
```
The test output was:
Loading model from ~/models/Llama-3.1-8B-Instruct...
Model loaded in 12.3s
[niah_single_1] Sample 0: PASS | Expected: 1234567 | Got: : 1234567.<|eot_id|>
[niah_single_1] Sample 1: PASS | Expected: 2345678 | Got: : 2345678.<|eot_id|>
... (50 more lines) ...
```

### Good (Uses ~50 tokens):
```
## Test Results: niah_single_1 (Samples 0-49)

**Pass Rate**: 50 / 50 (100%)

All samples passed.
```

---

## 8. Token Savings Estimate

| Result Type | Raw Output | Structured | Savings |
|-------------|------------|------------|---------|
| 50-sample test | ~1000 tokens | ~100 tokens | 90% |
| Benchmark run | ~500 tokens | ~80 tokens | 84% |
| Build failure | ~2000 tokens | ~200 tokens | 90% |

---

## 9. Integration

This rule should be applied when:
1. Spawning agents via Task tool
2. Running background commands
3. Processing results from completed agents

Combine with `multi-gpu-debugging.md` for efficient parallel testing workflows.
