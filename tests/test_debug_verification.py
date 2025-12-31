"""
Test KV cache offload correctness using debug hooks.
Injects distinctive K/V values, verifies loaded tensors match expected patterns.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

from random import randint, seed
from typing import Dict, List
import torch
from torch import Tensor
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context

# Config
MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 32 * 1024
NUM_GPU_BLOCKS = 4
INPUT_LEN = 32 * 1024
BLOCK_SIZE = 1024

# State
load_log: List[Dict] = []
current_chunk: List[int] = [0]


def debug_load_hook(slot_idx: int, layer_id: int, cpu_block_id: int, k: Tensor, v: Tensor) -> None:
    """Record loaded tensor values for layer 0."""
    if layer_id != 0:
        return

    load_log.append({
        "chunk_idx": current_chunk[0],
        "cpu_block_id": cpu_block_id,
        "k_value": k.float().mean().item(),
    })


def make_pattern_injection_hook(layer_id):
    """Inject K = chunk_idx + 1, V = -(chunk_idx + 1) for layer 0."""
    def hook(module, inputs):
        ctx = get_context()
        if not ctx.is_prefill or layer_id != 0:
            return inputs
        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
        current_chunk[0] = chunk_idx
        if len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            k_new = torch.full_like(k, float(chunk_idx + 1))
            v_new = torch.full_like(v, float(-(chunk_idx + 1)))
            return (q, k_new, v_new) + inputs[3:]
        return inputs
    return hook


def verify() -> bool:
    """Verify blocks loaded in correct order with correct K values."""
    chunk_loads: Dict[int, List[tuple]] = {}
    for log in load_log:
        chunk = log["chunk_idx"]
        if chunk not in chunk_loads:
            chunk_loads[chunk] = []
        chunk_loads[chunk].append((log["cpu_block_id"], log["k_value"]))

    for chunk, loads in chunk_loads.items():
        expected_blocks = list(range(chunk))
        actual_blocks = [b for b, _ in loads]
        k_values = [k for _, k in loads]
        expected_k = [float(b + 1) for b in expected_blocks]

        if actual_blocks != expected_blocks:
            return False
        if not all(abs(a - e) < 1e-2 for a, e in zip(k_values, expected_k)):
            return False
    return True


# Main
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

offload_engine = llm.model_runner.kvcache_manager.offload_engine
offload_engine.enable_debug_mode()
offload_engine.register_debug_hook(debug_load_hook)

hooks = []
for layer_idx, decoder_layer in enumerate(llm.model_runner.model.model.layers):
    hooks.append(decoder_layer.self_attn.attn.register_forward_pre_hook(
        make_pattern_injection_hook(layer_idx)
    ))

seed(42)
prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]
outputs = llm.generate(prompt_token_ids, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1), use_tqdm=False)

for hook in hooks:
    hook.remove()
offload_engine.remove_debug_hook(debug_load_hook)
offload_engine.disable_debug_mode()

# Verify
num_chunks = INPUT_LEN // BLOCK_SIZE
expected_loads = num_chunks * (num_chunks - 1) // 2
passed = len(load_log) == expected_loads and verify()

print(f"test_debug_verification: {'PASSED' if passed else 'FAILED'}")
