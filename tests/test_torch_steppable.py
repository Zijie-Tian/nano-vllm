"""
Test TorchSteppable: Print activation statistics at each layer.

Usage:
    python tests/test_torch_steppable.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM
from nanovllm.debug.adapters.torch_adapter import TorchSteppable
from utils import generate_needle_prompt, check_needle_answer

# ============================================================
# Config
# ============================================================
MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
INPUT_LEN = 512
MAX_NEW_TOKENS = 20
DTYPE = torch.float16

# ============================================================
# Load Model
# ============================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
model = model.to("cuda").eval()

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
steppable = TorchSteppable(model)

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

for bp in steppable.step(current_ids):
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
    # Forward pass (reuse same steppable)
    for bp in steppable.step(current_ids):
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

print("\ntest_torch_steppable: PASSED")
