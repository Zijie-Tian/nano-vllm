# Multi-GPU Debugging and Experimentation Rules

## Purpose

This rule governs GPU resource allocation and task execution strategy during debugging and experimentation on multi-GPU machines. The goal is to maximize debugging efficiency by:
- Running long validations on minimal GPUs (1-2)
- Using remaining GPUs for parallel hypothesis exploration
- Executing only one task/dataset for full validation during debugging

---

## 1. Scenario Classification

### 1.1 Long-Running Validation (Triggers Conservative Allocation)

A task SHALL be classified as **long-running validation** if ANY of the following conditions apply:

| Condition | Threshold |
|-----------|-----------|
| Estimated runtime | > 20 minutes |
| Sample count | > 50 samples per task |
| Full dataset execution | Any complete validation.jsonl |
| Full training/fine-tuning | Any training run |
| Large-scale inference | > 10K tokens total |

**Examples:**
- Running all 100 samples of `niah_single_1`
- Full RULER benchmark (13 tasks × 100 samples)
- Complete model evaluation on any benchmark

### 1.2 Exploratory / Fast-Iteration Work (Allows Full GPU Use)

A task SHALL be classified as **exploratory** if ALL of the following apply:

| Condition | Threshold |
|-----------|-----------|
| Estimated runtime | < 10 minutes |
| Sample count | ≤ 10 samples |
| Purpose | Sanity check, minimal reproduction, hypothesis testing |

**Examples:**
- Testing 3-5 specific error samples
- Single-batch inference for debugging
- Verifying a code fix on minimal input
- Profiling a single forward pass

---

## 2. GPU Allocation Strategy

### 2.1 Core Allocation Rules

| Task Type | GPU Allocation | Remaining GPUs |
|-----------|----------------|----------------|
| Long-running validation | 1 GPU (default), max 2 GPUs | Reserved for exploration |
| Exploratory work | As needed, can use multiple | - |

### 2.2 Mandatory Constraints

1. **MUST NOT** occupy all available GPUs for a single long-running validation
2. **MUST** reserve at least 50% of GPUs (minimum 2) for parallel exploration when ≥4 GPUs available
3. **MUST** select GPUs based on this priority:
   - Idle GPUs first (check with `nvidia-smi`)
   - If load info unavailable, use lowest-numbered GPUs for validation
4. **MUST** avoid resource conflicts:
   - Each task uses unique `CUDA_VISIBLE_DEVICES`
   - Each task uses unique output directories
   - Log files include GPU ID in filename

### 2.3 GPU Selection Algorithm

```
IF num_available_gpus >= 4:
    validation_gpus = 1 (or 2 if justified)
    exploration_gpus = remaining GPUs
ELSE IF num_available_gpus == 3:
    validation_gpus = 1
    exploration_gpus = 2
ELSE IF num_available_gpus == 2:
    validation_gpus = 1
    exploration_gpus = 1
ELSE:
    validation_gpus = 1
    exploration_gpus = 0 (sequential exploration)
```

---

## 3. Task / Dataset Selection Policy

### 3.1 Single-Task Validation Rule

During debugging, when a long-running validation is required:

- **MUST** execute only ONE task/dataset fully
- **MUST NOT** run all tasks unless explicitly requested or conditions in Section 4 are met

### 3.2 Task Selection Priority

Select the single task based on this priority order:

| Priority | Criterion | Example |
|----------|-----------|---------|
| 1 | Task most likely to reproduce the bug | If error occurs in `niah_single_1`, use that |
| 2 | Smallest task covering critical paths | `niah_single_1` (100 samples) vs `niah_multikey_3` |
| 3 | Task with known error samples | Use task with documented failure cases |
| 4 | Most representative task | Single-key before multi-key for basic validation |

### 3.3 Other Tasks Handling

Tasks not selected for full validation:
- **MAY** receive lightweight sanity checks (≤5 samples)
- **MUST NOT** receive full end-to-end execution by default
- **SHOULD** be noted in execution plan for future validation

---

## 4. Scale-Up Conditions

Expansion to more GPUs or multiple full tasks is **ALLOWED ONLY IF**:

| Condition | Justification Required |
|-----------|------------------------|
| Single-task validation completed successfully | Confirm fix works on one task first |
| Critical bug identified and fixed | Need cross-task verification |
| Cross-dataset consistency required | Clear technical justification needed |
| User explicitly requests full-scale | User override |

### 4.1 Default Behavior

- **DEFAULT**: Conservative, non-expansive
- **MUST** ask for confirmation before scaling up
- **MUST** document reason for scale-up in execution plan

---

## 5. Execution Plan Transparency

### 5.1 Mandatory Pre-Execution Output

Before starting any validation, **MUST** output an execution plan containing:

```markdown
## Execution Plan

### Task Classification
- Type: [Long-running validation / Exploratory]
- Reason: [Why classified this way]

### GPU Allocation
- Validation GPU(s): [GPU IDs]
- Reason: [Why these GPUs selected]
- Exploration GPU(s): [GPU IDs]
- Exploration tasks: [List of parallel hypotheses to test]

### Task Selection
- Full validation task: [Task name]
- Reason: [Why this task selected]
- Other tasks: [Skipped / Sanity-check only]

### Stopping Criteria
- Time limit: [X minutes]
- Success metric: [e.g., accuracy > 90%]
- Error threshold: [e.g., stop if >20 samples fail]

### Expected Output
- [What results will be produced]
```

### 5.2 Progress Checkpoints

For long-running validations, **SHOULD** report progress at:
- 25% completion
- 50% completion
- 75% completion
- Final results

---

## 6. Configuration Defaults

### 6.1 Default Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `LONG_RUNNING_THRESHOLD_MINUTES` | 20 | Runtime threshold for classification |
| `LONG_RUNNING_SAMPLE_THRESHOLD` | 50 | Sample count threshold |
| `MAX_VALIDATION_GPUS` | 2 | Maximum GPUs for long validation |
| `MIN_EXPLORATION_GPUS` | 2 | Minimum GPUs reserved for exploration (when ≥4 available) |
| `EXPLORATION_SAMPLE_LIMIT` | 10 | Max samples for exploratory tests |
| `SANITY_CHECK_SAMPLES` | 5 | Samples for non-selected tasks |

### 6.2 User Override

Users can override defaults by specifying in their request:
- "Use all GPUs for validation"
- "Run all tasks"
- "Increase validation GPUs to N"

---

## 7. Async Monitoring (CRITICAL)

### 7.1 Non-Blocking Principle

**MUST NOT** block the main agent with `sleep` commands waiting for results:
- ❌ `sleep 300 && check_results` (blocks main agent)
- ✅ Launch background tasks, continue thinking, check periodically

### 7.2 Continuous GPU Utilization

**MUST** maximize GPU utilization:
- When an agent completes a task, immediately assign new work
- Use `run_in_background: true` for all long-running agents
- Check agent completion via system notifications, not polling

### 7.3 Monitoring Strategy

```
CORRECT PATTERN:
1. Launch agents in background with run_in_background: true
2. Continue analysis, planning, or hypothesis generation
3. When agent completion notification arrives, process results
4. Immediately assign new tasks to freed GPUs

WRONG PATTERN:
1. Launch agents
2. sleep 300  # BLOCKS EVERYTHING!
3. Check results
4. GPU sits idle during sleep
```

### 7.4 Between-Task Work

While waiting for agents, the main agent SHOULD:
- Analyze code for additional hypotheses
- Prepare next batch of tests
- Update documentation with interim findings
- Plan fix implementations based on emerging patterns

### 7.5 Idle GPU Utilization (CRITICAL)

**MUST** utilize idle GPUs for exploratory tests while waiting:

```
WRONG PATTERN:
1. Launch 2 agents on GPU 0-1
2. Wait for completion  ← GPU 2-5 sit idle!
3. Process results

CORRECT PATTERN:
1. Launch 2 agents on GPU 0-1 for main validation
2. IMMEDIATELY launch exploratory tests on GPU 2-5:
   - Test alternative configurations
   - Verify edge cases
   - Run sanity checks on other datasets
   - Profile performance bottlenecks
3. Continue spawning new tasks as GPUs become free
4. Process results as they arrive
```

**Idle GPU Detection**:
```bash
# Check which GPUs are free
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

**Exploratory Test Ideas** (when main validation is running):

| GPU State | Suggested Task |
|-----------|----------------|
| Idle during single-task validation | Test same task with different config |
| Idle after quick test completes | Run related task (e.g., multikey after single-key) |
| Idle during long benchmark | Run profiling or memory analysis |
| Multiple GPUs idle | Parallelize hypothesis testing |

**Anti-Pattern**:
- ❌ "I'll wait for the 100-sample test to finish before doing anything else"
- ✅ "While GPU 0-1 run the 100-sample test, I'll use GPU 2-5 to test configs X, Y, Z"

---

## 8. Code Modification Policy (CRITICAL)

### 8.1 Evidence-Before-Action Principle

**MUST NOT** modify code until sufficient evidence has been gathered:

| Phase | Action | Code Modification |
|-------|--------|-------------------|
| Hypothesis Formation | Identify potential causes | ❌ NO |
| Evidence Gathering | Run targeted tests | ❌ NO |
| Pattern Analysis | Analyze test results | ❌ NO |
| Root Cause Confirmation | Validate with multiple tests | ❌ NO |
| Solution Design | Design fix based on evidence | ❌ NO |
| **Implementation** | Apply targeted fix | ✅ YES |

### 8.2 Minimum Evidence Requirements

Before proposing ANY code modification:

1. **Reproducibility**: Bug must be reproducible with specific test cases
2. **Isolation**: Root cause must be isolated (not symptoms)
3. **Multiple Data Points**: At least 3 independent test runs confirming the issue
4. **Counter-Evidence**: Attempted to disprove the hypothesis
5. **Mechanism Understanding**: Clear understanding of WHY the bug occurs

### 8.3 Main Agent Behavior

The main agent **SHOULD**:
- Keep thinking and analyzing while background agents run tests
- Formulate and refine hypotheses based on incoming results
- Document findings in `findings.md` as evidence accumulates
- Wait for sufficient test coverage before proposing fixes

The main agent **MUST NOT**:
- Rush to modify code after seeing first failure
- Propose fixes based on speculation
- Change multiple things at once "just to be safe"
- Assume correlation implies causation

### 8.4 Evidence Documentation Template

Before any code modification, document in `findings.md`:

```markdown
## Proposed Fix: [Brief Description]

### Evidence Summary
- Test A: [Result] - supports/contradicts hypothesis
- Test B: [Result] - supports/contradicts hypothesis
- Test C: [Result] - supports/contradicts hypothesis

### Root Cause Analysis
- What: [Specific bug behavior]
- Where: [File:line or function]
- Why: [Mechanism explanation]
- Confidence: [High/Medium/Low]

### Alternative Explanations Ruled Out
1. [Alternative A]: Ruled out because [reason]
2. [Alternative B]: Ruled out because [reason]

### Proposed Change
- File: [path]
- Change: [description]
- Expected Impact: [what should improve]
```

### 8.5 Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| See error → immediately edit code | See error → gather more data → analyze → then edit |
| Fix based on single test failure | Reproduce failure 3+ times, understand pattern |
| Change code "to see what happens" | Form hypothesis first, design targeted experiment |
| Modify multiple files simultaneously | Isolate changes, verify each independently |
| Skip documentation of findings | Document every significant finding before changing code |

---

## 9. Example Scenario

### Setup
- **Machine**: 8 GPUs (GPU 0-7)
- **Task**: Debug RULER chunked attention 20% error rate
- **Available tasks**: 6 RULER tasks (niah_single_1/2/3, niah_multikey_1/2/3)
- **Estimated full validation time**: ~2 hours for all tasks

### Execution Plan Output

```markdown
## Execution Plan

### Task Classification
- Type: Long-running validation
- Reason: Full validation of 100 samples × 6 tasks would take ~2 hours

### GPU Allocation
- Validation GPU(s): GPU 0 (1 GPU)
- Reason: Single GPU sufficient for sequential 100-sample validation
- Exploration GPU(s): GPU 1, 2, 3, 4, 5, 6, 7 (7 GPUs)
- Exploration tasks:
  1. GPU 1: Test 2-slot vs 4-slot ring buffer on error samples
  2. GPU 2: Test N-way merge implementation
  3. GPU 3: Test LSE precision fix
  4. GPU 4: Profile merge accumulation error
  5. GPU 5: Test with ruler_64k dataset (5 samples)
  6. GPU 6: Test decode boundary conditions
  7. GPU 7: Reserved for ad-hoc hypothesis testing

### Task Selection
- Full validation task: niah_single_1
- Reason: Has documented error samples (19 known failures), smallest single-key task
- Other tasks: Sanity-check only (5 samples each) after fix verified

### Stopping Criteria
- Time limit: 60 minutes for full validation
- Success metric: Error rate < 10% (down from 20%)
- Error threshold: Pause if new error pattern emerges (>5 consecutive failures)

### Expected Output
- Accuracy comparison: before vs after fix
- Error sample analysis: which samples still fail
- Hypothesis validation: which exploration branch identified the fix
```

### Execution Flow

1. **GPU 0**: Runs full `niah_single_1` validation (100 samples, ~40 min)
2. **GPU 1-7**: Run parallel exploration tasks (each ~5-15 min)
3. **Checkpoint at 50%**: Report GPU 0 progress + any discoveries from exploration
4. **On discovery**: If exploration GPU finds fix, pause validation, apply fix, restart
5. **Completion**: Report final results, decide if scale-up needed

---

## 10. Quick Reference Checklist

Before starting any debugging validation:

- [ ] Classified task type? (Long-running vs Exploratory)
- [ ] If long-running: Limited to 1-2 GPUs?
- [ ] If long-running: Selected single task for full validation?
- [ ] Remaining GPUs allocated for exploration?
- [ ] Execution plan output with all required sections?
- [ ] Stopping criteria defined?
- [ ] No user override requested? (Default conservative behavior)

Before proposing any code modification:

- [ ] Bug reproducible with specific test cases?
- [ ] Root cause isolated (not just symptoms)?
- [ ] At least 3 independent test runs confirming the issue?
- [ ] Alternative explanations ruled out?
- [ ] Mechanism of bug clearly understood?
- [ ] Evidence documented in findings.md?

---

## 11. Rule Violations

The following actions **VIOLATE** this rule:

1. Using all 6+ GPUs for a single 100-sample validation
2. Running full validation on all tasks without completing single-task first
3. Starting long validation without outputting execution plan
4. Not reserving GPUs for exploration when ≥4 GPUs available
5. Scaling up without meeting conditions in Section 4
6. **Modifying code before gathering sufficient evidence** (Section 8)
7. Proposing fixes based on single test failure or speculation
8. Changing multiple code locations simultaneously without isolation testing

---

## 12. Integration with Other Rules

This rule works alongside:
- `gpu-testing.md`: GPU type detection and basic allocation
- `planning-with-files.md`: Progress tracking for long validations
- `testing.md`: Test script conventions

When conflicts arise, this rule takes precedence for debugging scenarios.
