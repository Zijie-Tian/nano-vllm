"""
Test script for verifying KV cache offload correctness using debug hooks.

Strategy:
1. Inject distinctive K/V values (K=chunk_idx+1, V=-(chunk_idx+1))
2. Register debug hook to receive loaded tensor
3. Hook reads tensor values to verify correct block was loaded
4. No verification logic in framework - all external

This tests the framework's normal async execution path.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

from random import randint, seed
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 32 * 1024
NUM_GPU_BLOCKS = 4
INPUT_LEN = 32 * 1024
BLOCK_SIZE = 1024


# ============================================================
# External state (managed by test, not framework)
# ============================================================

# Record all load operations: list of {cpu_block_id, k_value, v_value, ...}
load_log: List[Dict] = []

# Track current chunk for grouping loads
current_chunk: List[int] = [0]  # mutable container


# ============================================================
# Debug hook - receives loaded tensor directly
# ============================================================

def debug_load_hook(slot_idx: int, layer_id: int, cpu_block_id: int, k: Tensor, v: Tensor) -> None:
    """
    Debug hook called after each H2D load.
    Reads tensor values to verify which block was actually loaded.
    """
    # Only record layer 0 for efficiency
    if layer_id != 0:
        return

    # Read tensor values (the distinctive pattern we injected)
    k_val = k.float().mean().item()
    v_val = v.float().mean().item()

    load_log.append({
        "chunk_idx": current_chunk[0],
        "slot_idx": slot_idx,
        "cpu_block_id": cpu_block_id,
        "k_value": k_val,
        "v_value": v_val,
    })


# ============================================================
# Pattern injection hook - injects distinctive values into K/V
# ============================================================

def make_pattern_injection_hook(layer_id):
    """Inject distinctive patterns: K = chunk_idx + 1, V = -(chunk_idx + 1)"""
    def hook(module, inputs):
        ctx = get_context()
        if not ctx.is_prefill:
            return inputs

        if layer_id != 0:
            return inputs

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        current_chunk[0] = chunk_idx  # Update for debug_load_hook

        if len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            k_pattern = float(chunk_idx + 1)
            v_pattern = float(-(chunk_idx + 1))
            k_new = torch.full_like(k, k_pattern)
            v_new = torch.full_like(v, v_pattern)
            return (q, k_new, v_new) + inputs[3:]

        return inputs
    return hook


# ============================================================
# Verification functions (all external, not in framework)
# ============================================================

def verify_load_order() -> Tuple[int, int, List[Dict]]:
    """Verify blocks were loaded in correct order by checking K values."""
    # Group loads by chunk
    chunk_loads: Dict[int, List[Tuple[int, float]]] = {}
    for log in load_log:
        chunk = log["chunk_idx"]
        if chunk not in chunk_loads:
            chunk_loads[chunk] = []
        chunk_loads[chunk].append((log["cpu_block_id"], log["k_value"]))

    correct = 0
    incorrect = 0
    errors = []

    for chunk in sorted(chunk_loads.keys()):
        loads = chunk_loads[chunk]
        # Expected: blocks [0, 1, ..., chunk-1] with K values [1, 2, ..., chunk]
        expected_blocks = list(range(chunk))
        actual_blocks = [block_id for block_id, _ in loads]

        # Also verify K values match expected pattern
        k_values = [k_val for _, k_val in loads]
        expected_k_values = [float(b + 1) for b in expected_blocks]

        blocks_ok = actual_blocks == expected_blocks
        # Check K values with tolerance
        k_ok = all(abs(a - e) < 1e-2 for a, e in zip(k_values, expected_k_values)) if len(k_values) == len(expected_k_values) else False

        if blocks_ok and k_ok:
            correct += 1
        else:
            incorrect += 1
            errors.append({
                "chunk_idx": chunk,
                "expected_blocks": expected_blocks,
                "actual_blocks": actual_blocks,
                "expected_k": expected_k_values,
                "actual_k": k_values,
            })

    return correct, incorrect, errors


def print_verification_summary():
    """Print verification results."""
    correct, incorrect, errors = verify_load_order()

    # Group for display
    chunk_loads: Dict[int, List[int]] = {}
    for log in load_log:
        chunk = log["chunk_idx"]
        if chunk not in chunk_loads:
            chunk_loads[chunk] = []
        chunk_loads[chunk].append(log["cpu_block_id"])

    print(f"\n{'='*60}")
    print("Debug Verification Summary")
    print(f"{'='*60}")

    print(f"\n1. Load Operations:")
    print(f"   Total H2D loads recorded: {len(load_log)}")
    print(f"   Chunks with correct order: {correct}")
    print(f"   Chunks with incorrect order: {incorrect}")

    if incorrect > 0:
        print(f"\n   Errors:")
        for err in errors[:5]:
            print(f"     Chunk {err['chunk_idx']}:")
            print(f"       Expected blocks: {err['expected_blocks']}")
            print(f"       Actual blocks:   {err['actual_blocks']}")
            print(f"       K values:        {[f'{v:.1f}' for v in err['actual_k']]}")

    print(f"\n2. Load Order Sample (first 5 and last 2 chunks):")
    sorted_chunks = sorted(chunk_loads.keys())
    display_chunks = sorted_chunks[:5] + sorted_chunks[-2:] if len(sorted_chunks) > 7 else sorted_chunks
    for chunk in display_chunks:
        blocks = chunk_loads[chunk]
        expected = list(range(chunk))
        status = "OK" if blocks == expected else "WRONG"
        print(f"   Chunk {chunk}: {blocks} [{status}]")

    print(f"\n{'='*60}")


# ============================================================
# Main Test Script
# ============================================================

print("Initializing LLM with CPU offload...")
llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_MODEL_LEN,
    enable_cpu_offload=True,
    kvcache_block_size=BLOCK_SIZE,
    num_gpu_blocks=NUM_GPU_BLOCKS,
    dtype="float16",
)

# Get offload engine and enable debug mode
kvcache_manager = llm.model_runner.kvcache_manager
offload_engine = kvcache_manager.offload_engine
offload_engine.enable_debug_mode()

# Register our debug hook
offload_engine.register_debug_hook(debug_load_hook)
print("Debug mode enabled with custom hook")

# Register pattern injection hooks
hooks = []
model = llm.model_runner.model
for layer_idx, decoder_layer in enumerate(model.model.layers):
    attn_module = decoder_layer.self_attn.attn
    pre_hook = attn_module.register_forward_pre_hook(make_pattern_injection_hook(layer_idx))
    hooks.append(pre_hook)
print(f"Registered {len(hooks)} pattern injection hooks")

# Generate input
seed(42)
prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]
num_chunks = INPUT_LEN // BLOCK_SIZE
print(f"\nInput: {INPUT_LEN} tokens, {num_chunks} chunks expected")
print(f"GPU blocks: {NUM_GPU_BLOCKS}, Block size: {BLOCK_SIZE}")

# Run prefill
print("\n" + "=" * 60)
print("Starting Prefill...")
print("=" * 60)

sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

# Remove hooks
for hook in hooks:
    hook.remove()
offload_engine.remove_debug_hook(debug_load_hook)

# Verify and print
print("\n" + "=" * 60)
print("Post-Execution Verification")
print("=" * 60)
print_verification_summary()

# Final verdict
correct, incorrect, _ = verify_load_order()
expected_loads = num_chunks * (num_chunks - 1) // 2
actual_loads = len(load_log)

print(f"\nResults:")
print(f"  Total loads: {actual_loads} (expected: {expected_loads})")
print(f"  Order verification: {correct} correct, {incorrect} incorrect")

print("\n" + "=" * 60)
all_passed = incorrect == 0 and actual_loads == expected_loads

if all_passed:
    print("test_debug_verification: PASSED")
else:
    print("test_debug_verification: FAILED")
print("=" * 60)

offload_engine.disable_debug_mode()
