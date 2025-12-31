"""
Correctness test for chunked prefill attention.

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
from flash_attn.flash_attn_interface import flash_attn_varlen_func

# Config
MODEL_PATH = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
MAX_MODEL_LEN = 128 * 1024
NUM_GPU_BLOCKS = 2
INPUT_LEN = 16 * 1024
BLOCK_SIZE = 1024

# State - capture Q and output for each (layer, chunk)
captures: List[Dict] = []


def make_ones_injection_hook():
    """Inject Q=K=V=1.0 for deterministic testing."""
    def hook(module, inputs):
        ctx = get_context()
        if not ctx.is_prefill:
            return inputs

        q, k, v = inputs[0], inputs[1], inputs[2]
        q_ones = torch.ones_like(q)
        k_ones = torch.ones_like(k)
        v_ones = torch.ones_like(v)
        return (q_ones, k_ones, v_ones) + inputs[3:]
    return hook


def make_capture_hook(layer_id: int):
    """Capture Q and output during prefill."""
    def hook(module, inputs, output):
        ctx = get_context()
        if not ctx.is_prefill:
            return

        q, k, v = inputs
        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0

        captures.append({
            'layer_id': layer_id,
            'chunk_idx': chunk_idx,
            'q': q.clone().cpu(),
            'k': k.clone().cpu(),
            'v': v.clone().cpu(),
            'output': output.clone().cpu(),
        })
    return hook


def compute_reference(layer_id: int, chunk_idx: int, scale: float,
                      k_cache_cpu: torch.Tensor, v_cache_cpu: torch.Tensor,
                      block_size: int) -> torch.Tensor:
    """
    Compute reference output using CPU KV cache and standard flash attention.

    Concatenates all Q, K, V from chunks 0..chunk_idx and runs causal attention,
    then extracts output for the current chunk.
    """
    # Get all captures for this layer up to chunk_idx
    layer_captures = [c for c in captures
                      if c['layer_id'] == layer_id and c['chunk_idx'] <= chunk_idx]
    layer_captures = sorted(layer_captures, key=lambda x: x['chunk_idx'])

    if not layer_captures:
        return None

    # Collect Q from captures, K/V from CPU cache
    all_q = []
    all_k = []
    all_v = []
    chunk_lengths = []

    for c in layer_captures:
        cidx = c['chunk_idx']
        q = c['q'].cuda()  # [seqlen, nheads, headdim]
        all_q.append(q)
        chunk_lengths.append(q.shape[0])

        # Get K, V from CPU cache (already offloaded during prefill)
        # CPU cache shape: [num_layers, num_blocks, block_size, kv_heads, head_dim]
        k = k_cache_cpu[layer_id, cidx, :q.shape[0]].cuda()
        v = v_cache_cpu[layer_id, cidx, :q.shape[0]].cuda()
        all_k.append(k)
        all_v.append(v)

    # Concatenate
    full_q = torch.cat(all_q, dim=0)
    full_k = torch.cat(all_k, dim=0)
    full_v = torch.cat(all_v, dim=0)
    total_len = full_q.shape[0]

    # Run standard causal flash attention
    cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32, device='cuda')
    full_o = flash_attn_varlen_func(
        full_q, full_k, full_v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=total_len,
        max_seqlen_k=total_len,
        softmax_scale=scale,
        causal=True,
    )

    # Extract output for current chunk
    start_pos = sum(chunk_lengths[:-1])
    end_pos = sum(chunk_lengths)
    return full_o[start_pos:end_pos].cpu()


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
outputs = llm.generate(prompt_token_ids, SamplingParams(temperature=0.6, max_tokens=1), use_tqdm=False)

# Remove hooks
for hook in hooks:
    hook.remove()

# Get CPU cache reference
offload_engine = llm.model_runner.kvcache_manager.offload_engine
k_cache_cpu = offload_engine.k_cache_cpu.clone()
v_cache_cpu = offload_engine.v_cache_cpu.clone()

# Verify: compare actual output with reference computed from CPU cache
all_passed = True
num_chunks = INPUT_LEN // BLOCK_SIZE

for idx,c in enumerate(captures):
    layer_id = c['layer_id']
    chunk_idx = c['chunk_idx']

    # Skip chunk 0 (no previous KV to load)
    if chunk_idx == 0:
        continue

    ref_output = compute_reference(layer_id, chunk_idx, scale, k_cache_cpu, v_cache_cpu, BLOCK_SIZE)
    if ref_output is None:
        continue

    actual_output = c['output']
    diff = (actual_output - ref_output).abs()
    max_diff = diff.max().item()
    
    passed = max_diff < 1e-1  # float16 tolerance
    all_passed = all_passed and passed

    if not passed:
        print(f"[FAIL] Layer {layer_id}, Chunk {chunk_idx}: max_diff={max_diff:.6f}")
        __import__('pdb').set_trace()

print(f"test_chunked_prefill_hook: {'PASSED' if all_passed else 'FAILED'}")
