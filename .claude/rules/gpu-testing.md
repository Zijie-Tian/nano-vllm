# GPU Testing Rules

## GPU Type Detection

Before running any GPU test/benchmark, detect the GPU type and apply appropriate settings:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

### Testing Mode by GPU Type

| GPU Type | Memory | Recommendation |
|----------|--------|----------------|
| **RTX 3090** | 24GB | May need reduced batch size |
| **RTX 4090** | 24GB | May need reduced batch size |
| **A100-40GB** | 40GB | Full testing OK |
| **A100-80GB** | 80GB | Full testing OK |
| **H100** | 80GB | Full testing OK |
| **Other** | - | Ask user for VRAM capacity |

---

## GPU Card Assignment (CRITICAL)

### Multi-Instance Environment

This project may run with multiple Claude instances, each needing a dedicated GPU.

### MANDATORY RULE

**Before executing ANY GPU command:**

1. **Check if user specified GPU**: Look for user message like "use GPU 0" or "CUDA_VISIBLE_DEVICES=1"

2. **If user did NOT specify GPU**:
   - **STOP and ASK**: "Which GPU should I use? (e.g., 0, 1, 2, ...)"
   - **DO NOT assume or guess** the GPU number
   - **DO NOT proceed** until user confirms

3. **Always prefix GPU commands with `CUDA_VISIBLE_DEVICES=X`**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python script.py  # Use GPU 0
   CUDA_VISIBLE_DEVICES=1 python script.py  # Use GPU 1
   ```

### Example Workflow

**Correct:**
```
User: "Run the RULER benchmark"
Claude: "Which GPU should I use for this benchmark?"
User: "Use GPU 2"
Claude: Runs `CUDA_VISIBLE_DEVICES=2 ./scripts/run_ruler.sh ...`
```

**Wrong:**
```
User: "Run the RULER benchmark"
Claude: Runs `./scripts/run_ruler.sh ...`  # NO! Missing GPU specification!
```

---

## Combined Checklist

Before running any GPU test:

- [ ] User specified GPU number? If not, ASK.
- [ ] Detected GPU type and VRAM capacity?
- [ ] Command prefixed with `CUDA_VISIBLE_DEVICES=X`?
- [ ] PYTHONPATH set correctly?
- [ ] Conda environment activated (`ruler`)?

## Example Commands

```bash
# Run RULER with GPU 0
CUDA_VISIBLE_DEVICES=0 ./scripts/run_ruler.sh llama3.1-8b-chat synthetic full

# Run with GPU 2
CUDA_VISIBLE_DEVICES=2 ./scripts/run_ruler.sh llama3.1-8b-chat synthetic xattn

# Multiple GPUs (if needed)
CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_ruler.sh llama3.1-8b-chat synthetic full
```

## Memory Estimation

For LLaMA 3.1 8B with different context lengths:

| Context Length | Approx. Memory (FP16) |
|----------------|----------------------|
| 4K | ~20GB |
| 8K | ~22GB |
| 16K | ~26GB |
| 32K | ~34GB |
| 64K | ~50GB |
| 128K | ~82GB |

Use sparse attention methods (`xattn`, `avgpool`, `compass`) for reduced memory on longer contexts.
