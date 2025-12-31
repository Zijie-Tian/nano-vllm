"""
Correctness test for chunked decode attention.

Captures Q and output during inference, then computes reference using
CPU KV cache with standard flash attention.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "WARNING"

import torch
from random import randint, seed
from typing import Dict, List
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context
from flash_attn.flash_attn_interface import flash_attn_func

# Config
MODEL_PATH = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
MAX_MODEL_LEN = 128 * 1024
NUM_GPU_BLOCKS = 2
INPUT_LEN = 16 * 1024
NUM_DECODE_TOKENS = 5
BLOCK_SIZE = 1024

# State
prefill_captures: List[Dict] = []
decode_captures: List[Dict] = []


def make_ones_injection_hook():
    """Inject Q=K=V=1.0 for deterministic testing."""
    def hook(module, inputs):
        q, k, v = inputs[0], inputs[1], inputs[2]
        q_ones = torch.ones_like(q)
        k_ones = torch.ones_like(k)
        v_ones = torch.ones_like(v)
        return (q_ones, k_ones, v_ones) + inputs[3:]
    return hook


def make_capture_hook(layer_id: int):
    """Capture Q, K, V, output during inference."""
    def hook(module, inputs, output):
        ctx = get_context()
        q, k, v = inputs

        if ctx.is_prefill:
            chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
            prefill_captures.append({
                'layer_id': layer_id,
                'chunk_idx': chunk_idx,
                'q': q.clone().cpu(),
                'k': k.clone().cpu(),
                'v': v.clone().cpu(),
                'output': output.clone().cpu(),
            })
        else:
            decode_step = len([c for c in decode_captures if c['layer_id'] == layer_id])
            decode_captures.append({
                'layer_id': layer_id,
                'decode_step': decode_step,
                'q': q.clone().cpu(),
                'k': k.clone().cpu(),
                'v': v.clone().cpu(),
                'output': output.clone().cpu(),
            })
    return hook


def compute_decode_reference(layer_id: int, decode_step: int, scale: float,
                              k_cache_cpu: torch.Tensor, v_cache_cpu: torch.Tensor,
                              block_size: int, num_prefill_chunks: int) -> torch.Tensor:
    """
    Compute reference decode output using CPU KV cache and standard flash attention.

    For decode, query attends to:
    1. All prefill KV (from CPU cache)
    2. All previous decode tokens (from captured decode k, v)
    """
    # Get decode capture for this layer and step
    decode_cap = None
    for c in decode_captures:
        if c['layer_id'] == layer_id and c['decode_step'] == decode_step:
            decode_cap = c
            break

    if decode_cap is None:
        return None

    # Query: single decode token
    q = decode_cap['q'].cuda()  # [1, num_heads, head_dim]
    q_batched = q.unsqueeze(0)  # [1, 1, num_heads, head_dim]

    # Collect all K, V: prefill chunks from CPU cache + decode tokens from captures
    all_k = []
    all_v = []

    # 1. Prefill chunks from CPU cache
    for cidx in range(num_prefill_chunks):
        # Get prefill capture to know the sequence length for this chunk
        prefill_cap = None
        for c in prefill_captures:
            if c['layer_id'] == layer_id and c['chunk_idx'] == cidx:
                prefill_cap = c
                break

        if prefill_cap is not None:
            seq_len = prefill_cap['q'].shape[0]
            k = k_cache_cpu[layer_id, cidx, :seq_len].cuda()
            v = v_cache_cpu[layer_id, cidx, :seq_len].cuda()
            all_k.append(k)
            all_v.append(v)

    # 2. Decode tokens from captures (up to and including current step)
    for step in range(decode_step + 1):
        for c in decode_captures:
            if c['layer_id'] == layer_id and c['decode_step'] == step:
                all_k.append(c['k'].cuda())
                all_v.append(c['v'].cuda())
                break

    if not all_k:
        return None

    # Concatenate all K, V
    full_k = torch.cat(all_k, dim=0).unsqueeze(0)  # [1, total_len, kv_heads, head_dim]
    full_v = torch.cat(all_v, dim=0).unsqueeze(0)

    # Run flash attention (non-causal since we explicitly control what KV to include)
    output = flash_attn_func(
        q_batched, full_k, full_v,
        softmax_scale=scale,
        causal=False,
    )

    return output.squeeze(0).squeeze(0).cpu()  # [num_heads, head_dim]


# ============================================================
# Main
# ============================================================

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

# Get model info
num_layers = len(llm.model_runner.model.model.layers)
head_dim = llm.model_runner.model.model.layers[0].self_attn.attn.head_dim
scale = head_dim ** -0.5

# Register hooks
hooks = []
for layer_idx, decoder_layer in enumerate(llm.model_runner.model.model.layers):
    # Pre-hook: inject all ones for Q, K, V
    # pre_hook = decoder_layer.self_attn.attn.register_forward_pre_hook(make_ones_injection_hook())
    # hooks.append(pre_hook)
    # Post-hook: capture Q, K, V, output
    post_hook = decoder_layer.self_attn.attn.register_forward_hook(make_capture_hook(layer_idx))
    hooks.append(post_hook)

# Run inference
seed(42)
prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]
outputs = llm.generate(prompt_token_ids, SamplingParams(temperature=0.6, max_tokens=NUM_DECODE_TOKENS), use_tqdm=False)

# Remove hooks
for hook in hooks:
    hook.remove()

# Get CPU cache reference
offload_engine = llm.model_runner.kvcache_manager.offload_engine
k_cache_cpu = offload_engine.k_cache_cpu.clone()
v_cache_cpu = offload_engine.v_cache_cpu.clone()

# Calculate number of prefill chunks
num_prefill_chunks = INPUT_LEN // BLOCK_SIZE

# Verify decode outputs
all_passed = True

for c in decode_captures:
    layer_id = c['layer_id']
    decode_step = c['decode_step']

    ref_output = compute_decode_reference(
        layer_id, decode_step, scale,
        k_cache_cpu, v_cache_cpu, BLOCK_SIZE, num_prefill_chunks
    )
    if ref_output is None:
        continue

    actual_output = c['output'].squeeze(0)
    if actual_output.dim() == 3:
        actual_output = actual_output.squeeze(0)

    diff = (actual_output - ref_output).abs()
    max_diff = diff.max().item()

    passed = max_diff < 1e-1
    all_passed = all_passed and passed

    # if not passed:
    print(f"[FAIL] Layer {layer_id}, Decode {decode_step}: max_diff={max_diff:.6f}")

print(f"test_chunked_decode_hook: {'PASSED' if all_passed else 'FAILED'}")
