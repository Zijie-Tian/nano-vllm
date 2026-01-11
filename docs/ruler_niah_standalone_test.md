# RULER NIAH Standalone Test Plan

## Overview

This document describes how to independently test nano-vllm's CPU offload functionality using RULER benchmark's NIAH (Needle-In-A-Haystack) task data.

## Background

### Problem Being Investigated

When running 32K sequence length tests with CPU offload mode, the model outputs garbled text instead of finding the magic number. This issue was traced to:

- **Root Cause**: Ring buffer `max_seq_len` was set equal to `max_model_len` (32768)
- **Issue**: When prefill uses ~32K tokens, decode needs to store KV at position 32768+, but ring buffer only has indices 0-32767
- **Fix Applied**: In `nanovllm/kvcache/__init__.py`, changed `max_seq_len = max_model_len + 512`

### Test Objective

Verify that the fix works correctly by running a standalone test with actual RULER NIAH data.

## Step 1: Copy Test Data

### Source Location

```
/home/zijie/Code/x-attention/eval/RULER/scripts/benchmark_root/full_fuse_16_llama3.1-8b-chat/synthetic/32768/data/niah_single_1/validation.jsonl
```

### Data Format

Each line is a JSON object:

```json
{
  "index": 0,
  "input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nA special magic number is hidden within the following text...",
  "outputs": ["8930103"],
  "length": 32768
}
```

- `input`: Full prompt with Llama 3.1 chat template (~122K characters, ~30K tokens)
- `outputs`: Expected answer (the magic number to find)
- `length`: Target sequence length in tokens

### Copy Command

```bash
mkdir -p /home/zijie/Code/nano-vllm/tests/data/ruler_niah
cp /home/zijie/Code/x-attention/eval/RULER/scripts/benchmark_root/full_fuse_16_llama3.1-8b-chat/synthetic/32768/data/niah_single_1/validation.jsonl \
   /home/zijie/Code/nano-vllm/tests/data/ruler_niah/niah_single_1_32k.jsonl
```

## Step 2: Create Test Script

Create `/home/zijie/Code/nano-vllm/tests/test_ruler_niah_32k.py`:

```python
"""
Standalone test for RULER NIAH task with 32K context length.

This test verifies that CPU offload mode correctly handles long sequences
where prefill tokens approach max_model_len.

Usage:
    python tests/test_ruler_niah_32k.py
"""

import json
import torch
from pathlib import Path

from nanovllm import LLM
from nanovllm.config import SamplingParams

# Configuration
MODEL_PATH = "/data/models/Llama-3.1-8B-Instruct"
DATA_FILE = Path(__file__).parent / "data/ruler_niah/niah_single_1_32k.jsonl"
MAX_MODEL_LEN = 32768
MAX_NEW_TOKENS = 50

# CPU Offload Settings
ENABLE_CPU_OFFLOAD = True
NUM_GPU_BLOCKS = 4
BLOCK_SIZE = 1024


def load_test_sample(filepath: Path, index: int = 0) -> dict:
    """Load a single test sample from JSONL file."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise ValueError(f"Sample index {index} not found")


def test_niah_single():
    """Test NIAH single needle task with 32K context."""
    print("=" * 60)
    print("RULER NIAH 32K Standalone Test")
    print("=" * 60)

    # Load test data
    sample = load_test_sample(DATA_FILE, index=0)
    prompt = sample["input"]
    expected = sample["outputs"][0]

    print(f"Prompt length: {len(prompt)} characters")
    print(f"Expected answer: {expected}")
    print()

    # Initialize model with CPU offload
    print("Initializing LLM with CPU offload...")
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        enable_cpu_offload=ENABLE_CPU_OFFLOAD,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        kvcache_block_size=BLOCK_SIZE,
        enforce_eager=True,  # Disable CUDA graphs for debugging
    )

    # Generate
    print("Generating response...")
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Expected: {expected}")
    print(f"Generated: {generated_text[:200]}...")
    print()

    # Check if expected number is in output
    if expected in generated_text:
        print("SUCCESS: Magic number found in output!")
        return True
    else:
        print("FAILED: Magic number NOT found in output")
        print(f"Full output: {generated_text}")
        return False


def test_multiple_samples(num_samples: int = 5):
    """Test multiple NIAH samples."""
    print("=" * 60)
    print(f"Testing {num_samples} NIAH samples with 32K context")
    print("=" * 60)

    # Initialize model once
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        enable_cpu_offload=ENABLE_CPU_OFFLOAD,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        kvcache_block_size=BLOCK_SIZE,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    correct = 0
    for i in range(num_samples):
        sample = load_test_sample(DATA_FILE, index=i)
        prompt = sample["input"]
        expected = sample["outputs"][0]

        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        if expected in generated_text:
            print(f"Sample {i}: PASS (found {expected})")
            correct += 1
        else:
            print(f"Sample {i}: FAIL (expected {expected}, got: {generated_text[:50]}...)")

    print()
    print(f"Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")
    return correct == num_samples


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        success = test_multiple_samples(5)
    else:
        success = test_niah_single()

    sys.exit(0 if success else 1)
```

## Step 3: Run Test

### Single Sample Test

```bash
cd /home/zijie/Code/nano-vllm
CUDA_VISIBLE_DEVICES=2,3,4,5 python tests/test_ruler_niah_32k.py
```

### All 5 Samples

```bash
cd /home/zijie/Code/nano-vllm
CUDA_VISIBLE_DEVICES=2,3,4,5 python tests/test_ruler_niah_32k.py --all
```

## Step 4: Expected Results

### Before Fix (Bug)

- Output: Garbled text like "not only has been replaced by thesiums..."
- Score: 0% (magic number not found)
- Time: ~80 seconds per sample

### After Fix (Expected)

- Output: The magic number (e.g., "8930103")
- Score: ~100% (magic number found)
- Time: ~80 seconds per sample (same, as the compute is unchanged)

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Ring Buffer Size

In the logs, verify:
```
OffloadEngine initializing: num_layers=32, num_kv_buffers=4, max_seq_len=33280
```

The `max_seq_len` should be `32768 + 512 = 33280` (not 32768).

### Monitor GPU Memory

```bash
watch -n 1 nvidia-smi
```

With CPU offload, GPU memory for KV cache should be ~640MB (ring buffer only).

## Related Files

| File | Description |
|------|-------------|
| `nanovllm/kvcache/__init__.py` | Fix location: `max_seq_len = max_model_len + 512` |
| `nanovllm/kvcache/offload_engine.py` | Ring buffer allocation |
| `nanovllm/engine/model_runner.py` | Layer-wise offload prefill/decode |
| `nanovllm/kvcache/hybrid_manager.py` | CPU block management |

## Test Data Details

### NIAH Task Description

The NIAH (Needle-In-A-Haystack) task tests the model's ability to retrieve a specific piece of information (the "needle") from a large context (the "haystack").

- **Needle**: A magic number associated with a keyword (e.g., "worried-purse")
- **Haystack**: ~30K tokens of distractor text
- **Task**: Extract the magic number when asked

### Sample Prompt Structure

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.

[... ~30K tokens of haystack text ...]

The special magic number for worried-purse is 8930103.

[... more haystack text ...]

What is the special magic number for worried-purse mentioned in the provided text?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

 The special magic number for worried-purse mentioned in the provided text is
```

The model should complete with: `8930103`
