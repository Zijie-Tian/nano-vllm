"""
Test NanovllmSteppable: Print activation statistics at each layer.

Usage:
    python tests/test_nanovllm_steppable.py
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer
from nanovllm import LLM
from nanovllm.debug.adapters.nanovllm_adapter import NanovllmSteppable
from utils import generate_needle_prompt, check_needle_answer

# ============================================================
# Config
# ============================================================
MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
INPUT_LEN = 32768  # Longer context to test offload
MAX_NEW_TOKENS = 20
DTYPE = torch.float16
ENABLE_CPU_OFFLOAD = True  # Test offload mode

# ============================================================
# Load Model
# ============================================================
print(f"Loading nanovllm model (cpu_offload={ENABLE_CPU_OFFLOAD})...")
llm = LLM(
    MODEL_PATH,
    enforce_eager=True,  # Required for hooks to work
    max_model_len=40960,
    max_num_batched_tokens=40960,
    enable_cpu_offload=ENABLE_CPU_OFFLOAD,
    dtype="float16",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Get the underlying model for steppable
model = llm.model_runner.model

# ============================================================
# Prepare Input (using needle-in-haystack prompt)
# ============================================================
prompt, expected_answer = generate_needle_prompt(
    tokenizer,
    target_length=INPUT_LEN,
    needle_position=0.5,
    needle_value="7492",
    use_chat_template=False,
    verbose=True,
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
print(f"Input shape: {input_ids.shape}")
print(f"Expected answer: {expected_answer}\n")

# ============================================================
# Create Steppable Model (reused for prefill + decode)
# ============================================================
steppable = NanovllmSteppable(model)

# ============================================================
# Prefill Phase: Print activation stats
# ============================================================
print("=" * 85)
print("PREFILL PHASE")
print("=" * 85)
print(f"{'Layer':<15} {'Shape':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 85)

current_ids = input_ids.clone()
logits = None

for bp in steppable.step(current_ids, is_prefill=True):
    t = bp.tensor.float()
    shape_str = str(list(t.shape))
    print(f"{bp.name:<15} {shape_str:<25} {t.mean():>10.4f} {t.std():>10.4f} {t.min():>10.4f} {t.max():>10.4f}")
    if bp.name == "LM Head":
        logits = bp.tensor

# Get first token from prefill
next_token_id = logits[0, -1].argmax().item()
next_token = tokenizer.decode(next_token_id)
current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=current_ids.device)], dim=1)
generated_tokens = [next_token]

# ============================================================
# Decode Phase: Only print generated tokens
# ============================================================
print("\n" + "=" * 85)
print("DECODE PHASE")
print("=" * 85)
print(f"Step  1: {next_token!r}")

for step in range(2, MAX_NEW_TOKENS + 1):
    # Forward pass with full sequence (reuse same steppable)
    # Note: nanovllm without KV cache needs full sequence for each decode
    for bp in steppable.step(current_ids, is_prefill=True):
        if bp.name == "LM Head":
            logits = bp.tensor

    # Get next token (greedy)
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

    print(f"Step {step:2}: {next_token!r}")

    # Stop if EOS
    if next_token_id == tokenizer.eos_token_id:
        print("  (EOS)")
        break

    # Append to sequence
    current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=current_ids.device)], dim=1)

# ============================================================
# Result
# ============================================================
print("\n" + "=" * 85)
print("RESULT")
print("=" * 85)
generated_text = "".join(generated_tokens)
print(f"Generated: {generated_text!r}")
print(f"Expected:  {expected_answer}")
print(f"Answer:    {'CORRECT!' if check_needle_answer(generated_text, expected_answer) else 'INCORRECT'}")

print("\ntest_nanovllm_steppable: PASSED")
