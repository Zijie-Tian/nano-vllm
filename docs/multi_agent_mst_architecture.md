# Multi-Agent MST Implementation Architecture

## Hardware Environment Specification

### Current Hardware
- **GPU**: NVIDIA A100 80GB (GPU 0)
- **CUDA Device**: cuda:0
- **Total Memory**: 80 GB
- **Test Target**: Simulate RTX 3090 24GB environment

### Mandatory GPU Selection
All tests and operations must explicitly use GPU 0:
```bash
CUDA_VISIBLE_DEVICES=0 python script.py
```

In code:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force use GPU 0
torch.cuda.set_device(0)
```

### Memory Monitoring Strategy
On A100 80GB, we will:
1. Monitor actual memory usage (0-80GB range)
2. Validate against 24GB RTX 3090 threshold
3. Ensure MST enables sequences that would OOM on 24GB

## Multi-Agent System Design

### Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Claude Flow Coordinator                    │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼─────┐ ┌───▼──────┐ ┌──▼──────────┐
│  Master     │ │  GPU     │ │  Validator  │
│  Agent      │ │  Monitor │ │  Agent      │
│             │ │  Agent   │ │             │
│ - Code Impl │ │ - Track  │ │ - Mem       │
│ - Integration│ │ - Report│ │ - Accuracy  │
│ - Testing   │ │ - Alert ││             │
└─────────────┘ └──────────┘ └─────────────┘
```

### Agent 1: Master Implementation Agent

**Responsibilities**:
1. Modify `config.py` - Add MST configuration
2. Modify `llama.py` - Add `forward_mst()` method
3. Modify `qwen3.py` - Add `forward_mst()` method
4. Modify `model_runner.py` - Integrate MST into prefill
5. Create test scripts
6. Execute initial tests

**Tools Access**:
- File read/edit/write
- Code search and navigation
- Test execution
- Git operations

**Communication**:
- Reports progress to Coordinator
- Receives GPU metrics from Monitor
- Receives validation results from Validator

### Agent 2: GPU Monitor Agent

**Responsibilities**:
1. Continuously monitor GPU 0 memory usage
2. Track peak memory during tests
3. Report memory consumption in real-time
4. Alert if approaching simulated 24GB limit
5. Generate memory usage graphs

**Implementation**:
```python
class GPUMonitorAgent:
    def __init__(self):
        self.device = torch.device('cuda:0')  # Force GPU 0
        self.target_limit = 24 * 1024 ** 3   # 24GB limit
        self.monitor_interval = 1.0          # Check every second

    def start_monitoring(self):
        """Start background monitoring thread"""
        while self.is_monitoring:
            memory_stats = torch.cuda.memory_stats()
            allocated = memory_stats['allocated_bytes.all.current']
            peak = torch.cuda.max_memory_allocated()

            # Report to coordinator
            self.send_update({
                'allocated_gb': allocated / 1e9,
                'peak_gb': peak / 1e9,
                'approaching_24gb_limit': peak > (0.9 * self.target_limit),
                'would_oom_on_3090': peak > self.target_limit,
            })

            time.sleep(self.monitor_interval)
```

**Alerts Generated**:
- 🟢 Normal operation: Peak MLP memory X GB (Y% of 24GB limit)
- 🟡 Approaching limit: Peak memory X GB (>90% of 24GB limit)
- 🔴 Critical: Peak memory X GB (would OOM on RTX 3090!)
- ✅ Success: Sequence would OOM but MST prevents it

### Agent 3: Validator Agent

**Responsibilities**:
1. **Memory Validation**:
   - Measure actual memory usage per experiment
   - Compare against theoretical calculations
   - Validate 16x reduction claim

2. **Accuracy Validation**:
   - Run needle test with MST enabled
   - Verify output equivalence (would_seq_oom, but_mst_prevents_it)
   - Check for numerical differences

3. **Threshold Monitoring**:
   - Verify sequences > 32k activate MST correctly
   - Sequences ≤ 32k should NOT use MST
   - Validate mst_min_seq_len parameter

**Test Matrix**:

| Seq Length | Without MST | With MST | Should Use MST? | Status |
|------------|-------------|----------|-----------------|--------|
| 16k | 18 GB | 18 GB | No (<32k) | ✅ |
| 32k | 19 GB | 19 GB | Yes (>32k) | ✅ |
| 64k | OOM | 20 GB | Yes | 🔑 Critical |
| 128k | OOM | 21 GB | Yes | 🔑 Critical |
| 256k | OOM | 45 GB | Yes | 🔑 Critical |

## Multi-Agent Workflow

### Workflow 1: Initial GPU Verification

```
Coordinator → GPU Monitor Agent:
"Check GPU status. Force use GPU 0 only."

GPU Monitor Agent → Coordinator:
"✅ GPU 0 detected: A100 80GB
   ⚡ 注意: Will simulate RTX 3090 24GB limit for testing"
```

### Workflow 2: Memory Baseline Measurement

```
Coordinator → GPU Monitor Agent + Master Agent:
"Run baseline test without MST for 64k sequence"

Master Agent:
- Execute: CUDA_VISIBLE_DEVICES=0 python test_baseline.py --seq-len 65536
- Monitor output

GPU Monitor Agent:
- Track memory: Peak X GB
- Alert: "🔴 Would OOM on RTX 3090! (X > 24GB)"

Validator Agent:
- Verify peak matches theoretical 21 GB
- Status: ✅ Baseline established
```

### Workflow 3: MST Implementation Verification

```
Coordinator → Master Agent:
"Implement Phase 1-3 (MLP modifications)"

Master Agent:
- Modify config.py, llama.py, qwen3.py, model_runner.py
- Report: "Phase complete, ready for testing"

Coordinator → Master + GPU Monitor + Validator:
"Run test with MST enabled for 64k sequence"

Master Agent:
- Execute: CUDA_VISIBLE_DEVICES=0 python test_mst_memory.py
- Stream logs to Coordinator

GPU Monitor Agent (continuous):
- "🟢 Allocated: 5.2 GB | Peak: 20.1 GB | 84% of 24GB limit"
- "✅ Would NOT OOM on RTX 3090 (20.1 < 24)"

Validator Agent:
- Verify 16x memory reduction (theoretical: 20.1 GB)
- Check accuracy maintained
- Report: "✅ MST successfully prevents OOM"
```

### Workflow 4: Long-Sequence Validation

```
Coordinator → All Agents:
"Test 128k and 256k sequences on GPU 0"

Master Agent:
- For each length: run inference test
- Record latency and throughput

GPU Monitor Agent:
- 128k: "🟢 Peak 21.3 GB (89% of 24GB) ✅ Would NOT OOM"
- 256k: "🟢 Peak 23.8 GB (99% of 24GB) ⚠️ Approaching limit"

Validator Agent:
- Confirm threshold: MST activated (>32k)
- Verify accuracy: Needle test at 100%
- Performance: 18% latency overhead (within 20% target)

Final Report:
"✅ 128k tokens: WORKS (would OOM without MST)
 ✅ 256k tokens: WORKS (临界, would OOM without MST)
 ✅ 16x memory reduction validated
 ✅ Accuracy maintained"
```

## Command Modifications for GPU 0

### All Commands Must Include
```bash
# Force GPU 0 only
CUDA_VISIBLE_DEVICES=0 python script.py

# In Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Test Command Updates

#### Memory Test
```bash
# Before
python tests/test_mst_memory.py

# After (mandatory)
CUDA_VISIBLE_DEVICES=0 python tests/test_mst_memory.py --gpu 0
```

#### Needle Test
```bash
# Before
python tests/test_needle.py --enable-offload --enable-mst

# After (mandatory)
CUDA_VISIBLE_DEVICES=0 python tests/test_needle.py --enable-offload --enable-mst --gpu 0
```

#### Memory Profiling Script
```python
# In test scripts, add at top
import os
import torch

# Force GPU 0 before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
```

## A100 80GB Testing Advantages

While simulating RTX 3090 24GB limit, testing on A100 provides:

1. **Safety Margin**: Can test sequences that exceed 24GB without actual system failure
2. **Monitoring Accuracy**: An 80GB - 24GB = 56GB monitoring window to catch memory spikes
3. **Baseline Collection**: Can measure "what if" scenarios without OOM
4. **Profiler Data**: NVidia profiling tools work better on A100

### Memory Monitoring Strategy

```python
# In validation agent
def validate_against_24gb_limit(self, actual_peak_gb):
    """Validate against RTX 3090 simulated limit"""
    limit_gb = 24.0

    if actual_peak_gb < limit_gb:
        return {
            'status': '✅ WOULDN\'T OOM',
            'peak': actual_peak_gb,
            'margin_gb': limit_gb - actual_peak_gb,
            'margin_percent': (limit_gb - actual_peak_gb) / limit_gb * 100,
        }
    else:
        return {
            'status': '🔴 WOULD OOM on RTX 3090!',
            'peak': actual_peak_gb,
            'overage_gb': actual_peak_gb - limit_gb,
        }

# Example outputs:
# - 19.2 GB: "✅ WOULDN'T OOM | Margin: 4.8 GB (20.0%)"
# - 25.1 GB: "🔴 WOULD OOM! | Overage: 1.1 GB"
```

## Agent Coordination Protocol

### Communication Flow

```
1. Coordinator initiates task
   → Assigns to Master Agent

2. Master Agent starts work
   → Sends "STARTING" status to Coordinator
   → GPU Monitor starts background monitoring

3. Master Agent executes steps
   → Reports progress to Coordinator
   → GPU Monitor reports memory continuously

4. Master Agent completes step
   → Sends "COMPLETED" with results
   → Validator verifies and reports

5. Coordinator collects all reports
   → Makes decision on next step
   → Updates task status
```

### Status Updates Format

**GPU Monitor Status**:
```json
{
  "timestamp": "2024-01-15T10:30:45",
  "gpu_id": 0,
  "allocated_gb": 15.2,
  "peak_gb": 20.1,
  "simulated_limit_gb": 24.0,
  "status": "✅ SAFE",
  "alert": null
}
```

**Validator Status**:
```json
{
  "timestamp": "2024-01-15T10:31:00",
  "test_type": "memory_reduction",
  "sequence_length": 65536,
  "expected_reduction": 16.0,
  "measured_reduction": 16.3,
  "threshold_violated": false,
  "accuracy_preserved": true,
  "status": "✅ PASS"
}
```

## Implementation Changes Summary

### Files to Modify (Master Agent)

1. **nanovllm/config.py**: Add MST configuration
2. **nanovllm/models/llama.py**: Add forward_mst()
3. **nanovllm/models/qwen3.py**: Add forward_mst()
4. **nanovllm/engine/model_runner.py**: Integrate MST logic
5. **tests/test_mst_gpu0.py**: GPU 0 specific tests

### Files to Create (All Agents)

1. **agents/gpu_monitor.py**: GPU monitoring agent
2. **agents/validator.py**: Validation agent
3. **tests/test_multiagent_coordination.py**: Coordination test
4. **docs/multiagent_setup.md**: Multi-agent setup guide

## Success Criteria with Multi-Agent

### Primary Goals
1. ✅ MST implementation complete
2. ✅ 128k tokens run on GPU 0 at ~21GB (< 24GB limit)
3. ✅ 16x memory reduction verified by Validator
4. ✅ 100% needle accuracy maintained

### Multi-Agent Specific Goals
1. ✅ GPU Monitor continuously tracks GPU 0
2. ✅ All commands explicitly use CUDA_VISIBLE_DEVICES=0
3. ✅ Validator reports against 24GB simulated limit
4. ✅ Agents coordinate successfully without conflicts

### Measurement Requirements

**Per-Experiment Tracking**:
```
Test: 128k_tokens_mst_enabled
├─ GPU Monitor Report:
│  ├─ Peak Memory: 21.3 GB
│  ├─ Would OOM on 3090: NO (21.3 < 24.0)
│  └─ Status: ✅ PASS
│
├─ Validator Report:
│  ├─ Memory Reduction: 16.2x
│  ├─ Accuracy: 100% (needle test)
│  └─ Status: ✅ PASS
│
└─ Master Agent Report:
   ├─ Implementation: Complete
   ├─ Test Executed: YES
   └─ Status: ✅ PASS
```

## Priority Order

### Phase 0: Setup (Before Implementation)
1. ✅ Confirm GPU 0 is A100 80GB
2. ✅ Update all plan documents with GPU 0 requirements
3. ✅ Design agent communication protocol
4. ✅ Create GPU monitoring script

### Phase 1-5: Implementation (Parallel Execution)
- Master Agent: Code modifications
- GPU Monitor: Continuous monitoring
- Validator: Intermittent verification

### Phase 6: Integration Testing
- All agents active
- Coordination verification
- Final report generation
