# GPU Testing Rules

## GPU Type Detection

Before running any GPU test/benchmark, detect the GPU type and apply appropriate settings:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

### Testing Mode by GPU Type

| GPU Type | Test Mode | Reason |
|----------|-----------|--------|
| **RTX 3090** | `--enable-offload` ONLY | Limited VRAM (24GB), must use CPU offload |
| **A100** | Both modes OK | Large VRAM (40/80GB), can test with or without offload |
| **RTX 4090** | `--enable-offload` ONLY | Limited VRAM (24GB) |
| **Other** | Ask user | Unknown VRAM capacity |

### Example Commands

**For 3090:**
```bash
# MUST use offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload
```

**For A100:**
```bash
# Can test without offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct

# Or with offload
CUDA_VISIBLE_DEVICES=X python tests/test_needle.py --model ~/models/Llama-3.1-8B-Instruct --enable-offload
```

---

## GPU Card Assignment (CRITICAL)

### Multi-Instance Environment

This project runs with multiple Claude instances on different worktrees, each needing a dedicated GPU.

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
User: "Run the needle test"
Claude: "Which GPU should I use for this test?"
User: "Use GPU 2"
Claude: Runs `CUDA_VISIBLE_DEVICES=2 python tests/test_needle.py ...`
```

**Wrong:**
```
User: "Run the needle test"
Claude: Runs `python tests/test_needle.py ...`  # NO! Missing GPU specification!
```

---

## Combined Checklist

Before running any GPU test:

- [ ] User specified GPU number? If not, ASK.
- [ ] Detected GPU type? (3090 → offload only, A100 → flexible)
- [ ] GPU mutex check passed? (see commands.md)
- [ ] Command prefixed with `CUDA_VISIBLE_DEVICES=X`?
- [ ] Local package installed? (`pip install -e . --prefix=./.local --no-deps`)
