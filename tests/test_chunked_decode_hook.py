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

    # Collect all K, V: prefill chunks from captures + decode tokens from captures
    # NOTE: We use prefill captures directly instead of CPU cache because
    # the CPU block ID may not equal the chunk index.
    all_k = []
    all_v = []

    # 1. Prefill chunks from captures (use captured K/V, not CPU cache)
    for cidx in range(num_prefill_chunks):
        prefill_cap = None
        for c in prefill_captures:
            if c['layer_id'] == layer_id and c['chunk_idx'] == cidx:
                prefill_cap = c
                break

        if prefill_cap is not None:
            # Use captured K/V directly (guaranteed to be correct layer data)
            all_k.append(prefill_cap['k'].cuda())
            all_v.append(prefill_cap['v'].cuda())

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

# Debug: Compare decode_buffer with captured K/V
print("\n=== DEBUG: Comparing decode_buffer with captured K/V ===")
decode_k_buffer = offload_engine.decode_k_buffer.clone().cpu()
for step in range(NUM_DECODE_TOKENS):
    for layer_id in [0, 17, 35]:  # Sample a few layers
        # Find captured K for this step and layer
        for c in decode_captures:
            if c['layer_id'] == layer_id and c['decode_step'] == step:
                captured_k = c['k'].squeeze(0)  # [kv_heads, head_dim]
                buffer_k = decode_k_buffer[layer_id, step]  # [kv_heads, head_dim]
                diff = (captured_k - buffer_k).abs().max().item()
                print(f"Step {step}, Layer {layer_id}: captured vs buffer max_diff={diff:.6f}")
                break

# Debug: Verify that decode_buffer slices match concatenated captures
print("\n=== DEBUG: Verifying decode_buffer slices ===")
for layer_id in [0]:
    for decode_step in [1, 2]:  # Check steps that use multiple tokens
        # Build expected slice from captures
        expected_k_list = []
        for step in range(decode_step + 1):
            for c in decode_captures:
                if c['layer_id'] == layer_id and c['decode_step'] == step:
                    expected_k_list.append(c['k'].squeeze(0))  # [kv_heads, head_dim]
                    break
        if expected_k_list:
            expected_k = torch.stack(expected_k_list, dim=0)  # [num_tokens, kv_heads, head_dim]
            buffer_slice = decode_k_buffer[layer_id, 0:decode_step+1]
            diff = (expected_k - buffer_slice).abs().max().item()
            print(f"Decode step {decode_step}, Layer {layer_id}: buffer slice vs expected max_diff={diff:.6f}")
            # Print first values
            print(f"  Buffer[0,0,0]={buffer_slice[0,0,0].item():.6f}, Expected[0,0,0]={expected_k[0,0,0].item():.6f}")
            if decode_step >= 1:
                print(f"  Buffer[1,0,0]={buffer_slice[1,0,0].item():.6f}, Expected[1,0,0]={expected_k[1,0,0].item():.6f}")

# Debug: Print expected K value for block 0, layer 0 (to compare with actual loading)
print("\n=== DEBUG: Expected K values for block 0, layer 0 ===")
for c in prefill_captures:
    if c['layer_id'] == 0 and c['chunk_idx'] == 0:
        print(f"Captured K[0,0,0] for layer 0, chunk 0: {c['k'][0,0,0].item():.6f}")
        break
print(f"CPU cache K[0,0,0,0,0] for layer 0, block 0: {k_cache_cpu[0,0,0,0,0].item():.6f}")

# Debug: Compare CPU cache with captured prefill K/V
print("\n=== DEBUG: Comparing CPU cache with captured prefill K/V ===")
for chunk_idx in [0, 7, 15]:  # Sample a few chunks
    for layer_id in [0, 17, 35]:  # Sample a few layers
        # Find captured K for this chunk and layer
        for c in prefill_captures:
            if c['layer_id'] == layer_id and c['chunk_idx'] == chunk_idx:
                captured_k = c['k']  # [seq_len, kv_heads, head_dim]
                cpu_cache_k = k_cache_cpu[layer_id, chunk_idx, :captured_k.shape[0]]
                diff = (captured_k - cpu_cache_k).abs().max().item()
                print(f"Chunk {chunk_idx}, Layer {layer_id}: captured vs CPU cache max_diff={diff:.6f}")
                break

# Debug: Get cpu_block_table to check order
kvcache_manager = llm.model_runner.kvcache_manager
# Find the sequence (it should still exist)
from nanovllm.engine.sequence import Sequence
for attr_name in ['sequences', '_sequences', 'active_sequences']:
    if hasattr(kvcache_manager, attr_name):
        print(f"Found {attr_name}")
        break

# Try to get cpu_block_table through a different way
print(f"\n=== DEBUG: CPU block order ===")
# For each prefill capture, check which CPU block it ended up in
for chunk_idx in range(num_prefill_chunks):
    for c in prefill_captures:
        if c['layer_id'] == 0 and c['chunk_idx'] == chunk_idx:
            # Check if this chunk's K matches any CPU block
            captured_k_first = c['k'][0, 0, 0].item()
            for block_id in range(num_prefill_chunks):
                cpu_k_first = k_cache_cpu[0, block_id, 0, 0, 0].item()
                if abs(captured_k_first - cpu_k_first) < 1e-6:
                    print(f"Chunk {chunk_idx} -> CPU block {block_id}")
                    break
            break

# Debug: Check reference vs actual for decode steps 0 and 1
# Also compute partial references (prefill only, decode only) to isolate the bug
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs
for decode_step in [0, 1]:
    print(f"\n=== DEBUG: Reference vs Actual for layer 0, decode {decode_step} ===")
    layer_id = 0
    # Find the capture
    for c in decode_captures:
        if c['layer_id'] == layer_id and c['decode_step'] == decode_step:
            q = c['q'].cuda()  # [1, num_heads, head_dim]
            q_batched = q.unsqueeze(0)  # [1, 1, num_heads, head_dim]

            # Build prefill K/V per-block for block-by-block reference
            prefill_k_blocks = []
            prefill_v_blocks = []
            for cidx in range(num_prefill_chunks):
                for pc in prefill_captures:
                    if pc['layer_id'] == layer_id and pc['chunk_idx'] == cidx:
                        prefill_k_blocks.append(pc['k'].cuda().unsqueeze(0))  # [1, block_size, kv_heads, head_dim]
                        prefill_v_blocks.append(pc['v'].cuda().unsqueeze(0))
                        break

            # Build decode K/V
            decode_k_list = []
            decode_v_list = []
            for step in range(decode_step + 1):
                for dc in decode_captures:
                    if dc['layer_id'] == layer_id and dc['decode_step'] == step:
                        decode_k_list.append(dc['k'].cuda())
                        decode_v_list.append(dc['v'].cuda())
                        break

            full_prefill_k = torch.cat([kb.squeeze(0) for kb in prefill_k_blocks], dim=0).unsqueeze(0)
            full_prefill_v = torch.cat([vb.squeeze(0) for vb in prefill_v_blocks], dim=0).unsqueeze(0)
            full_decode_k = torch.cat(decode_k_list, dim=0).unsqueeze(0)
            full_decode_v = torch.cat(decode_v_list, dim=0).unsqueeze(0)

            full_k = torch.cat([full_prefill_k, full_decode_k], dim=1)
            full_v = torch.cat([full_prefill_v, full_decode_v], dim=1)

            print(f"Q shape: {q_batched.shape}")
            print(f"Prefill K shape: {full_prefill_k.shape}")
            print(f"Decode K shape: {full_decode_k.shape}")
            print(f"Full K shape: {full_k.shape}")
            print(f"Total tokens: prefill={num_prefill_chunks * BLOCK_SIZE}, decode={decode_step + 1}")

            # Reference output (single attention over all)
            ref_output = flash_attn_func(
                q_batched, full_k, full_v,
                softmax_scale=scale,
                causal=False,
            )

            # Chunked reference: prefill attention + decode attention + merge
            prefill_o, prefill_lse = flash_attn_with_lse(
                q_batched, full_prefill_k, full_prefill_v,
                softmax_scale=scale,
                causal=False,
            )
            decode_o, decode_lse = flash_attn_with_lse(
                q_batched, full_decode_k, full_decode_v,
                softmax_scale=scale,
                causal=False,
            )
            chunked_output, _ = merge_attention_outputs(prefill_o, prefill_lse, decode_o, decode_lse)

            # Block-by-block reference (simulating ring buffer pipeline)
            block_o_acc, block_lse_acc = None, None
            for bidx, (kb, vb) in enumerate(zip(prefill_k_blocks, prefill_v_blocks)):
                o_blk, lse_blk = flash_attn_with_lse(q_batched, kb, vb, softmax_scale=scale, causal=False)
                if block_o_acc is None:
                    block_o_acc, block_lse_acc = o_blk, lse_blk
                else:
                    block_o_acc, block_lse_acc = merge_attention_outputs(block_o_acc, block_lse_acc, o_blk, lse_blk)

            # Compare block-by-block vs single
            block_vs_single_diff = (block_o_acc - prefill_o).abs().max().item()
            print(f"Block-by-block vs Single max_diff: {block_vs_single_diff:.6f}")

            # Compare full reference vs chunked reference
            ref_vs_chunked_diff = (ref_output - chunked_output).abs().max().item()
            print(f"Reference vs Chunked-reference max_diff: {ref_vs_chunked_diff:.6f}")

            ref_output = ref_output.squeeze(0).squeeze(0).cpu()
            chunked_output_cpu = chunked_output.squeeze(0).squeeze(0).cpu()

            # Actual output
            actual_output = c['output'].squeeze(0)
            if actual_output.dim() == 3:
                actual_output = actual_output.squeeze(0)

            diff_ref = (actual_output - ref_output).abs()
            diff_chunked = (actual_output - chunked_output_cpu).abs()
            print(f"Actual vs Reference max_diff: {diff_ref.max().item():.6f}")
            print(f"Actual vs Chunked-ref max_diff: {diff_chunked.max().item():.6f}")
            break
print()

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

    if not passed:
        print(f"[FAIL] Layer {layer_id}, Decode {decode_step}: max_diff={max_diff:.6f}")

print(f"test_chunked_decode_hook: {'PASSED' if all_passed else 'FAILED'}")
