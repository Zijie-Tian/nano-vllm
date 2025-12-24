"""
Test script for chunked prefill with CPU offload.

Demonstrates: LLM initialization, prefill execution with CPU offload enabled.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

from random import randint, seed
from nanovllm import LLM, SamplingParams


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 32 * 1024
NUM_GPU_BLOCKS = 2
INPUT_LEN = 16 * 1024

# ============================================================
# Main Test Script
# ============================================================

# 1. Initialize LLM with CPU offload
llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_MODEL_LEN,
    enable_cpu_offload=True,
    kvcache_block_size=1024,
    num_gpu_blocks=NUM_GPU_BLOCKS,
)

# 2. Generate random prompt tokens
seed(42)
prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]

# 3. Run prefill (max_tokens=1 to focus on prefill only)
sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

# 4. Verify output
assert len(outputs) == 1
assert "token_ids" in outputs[0]
assert len(outputs[0]["token_ids"]) == 1

print("test_prefill: PASSED")
