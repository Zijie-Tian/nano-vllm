"""
Hook-based correctness test for chunked decode attention.

Uses PyTorch register_forward_hook() to capture real inference I/O,
then compares against reference computation to locate bugs.

This test targets the decode phase with CPU offload - after prefill,
the model generates tokens one by one while attending to all previous context.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B/")
MAX_MODEL_LEN = 8 * 1024
NUM_GPU_BLOCKS = 2
INPUT_LEN = 2 * 1024  # 2K tokens for prefill
NUM_DECODE_TOKENS = 5  # Generate 5 tokens to test decode
BLOCK_SIZE = 1024


# ============================================================
# Global capture storage
# ============================================================

captures = []
prefill_kv = {}  # Store prefill k,v for reference computation


# ============================================================
# Hook Functions
# ============================================================

def make_hook(layer_id):
    """Create a forward hook for a specific layer."""
    def hook(module, inputs, output):
        q, k, v = inputs
        ctx = get_context()

        is_prefill = ctx.is_prefill

        capture_entry = {
            'layer_id': layer_id,
            'is_prefill': is_prefill,
            'q': q.clone().cpu(),
            'k': k.clone().cpu(),
            'v': v.clone().cpu(),
            'output': output.clone().cpu(),
            'is_chunked_prefill': ctx.is_chunked_prefill,
        }

        if is_prefill:
            # Store prefill k,v for reference computation
            chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0
            capture_entry['chunk_idx'] = chunk_idx
            if layer_id not in prefill_kv:
                prefill_kv[layer_id] = []
            prefill_kv[layer_id].append({
                'chunk_idx': chunk_idx,
                'k': k.clone().cpu(),
                'v': v.clone().cpu(),
            })
        else:
            # Decode phase - capture decode token info
            capture_entry['decode_step'] = len([c for c in captures
                                                 if c['layer_id'] == layer_id and not c['is_prefill']])

        captures.append(capture_entry)
    return hook


def register_hooks(llm):
    """Register forward hooks on all Attention modules."""
    hooks = []
    model = llm.model_runner.model

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn_module = decoder_layer.self_attn.attn
        hook = attn_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    return hooks


# ============================================================
# Reference Computation
# ============================================================

def compute_decode_reference(layer_id, decode_step, scale, debug=False):
    """
    Compute reference decode attention output for a specific layer.

    For decode, the query is a single token that attends to:
    1. All prefill KV (from CPU cache)
    2. All previous decode tokens (stored in GPU decode slot)
    """
    # Get the decode capture
    decode_captures = [c for c in captures
                       if c['layer_id'] == layer_id and not c['is_prefill']]
    if decode_step >= len(decode_captures):
        return None

    decode_capture = decode_captures[decode_step]
    q = decode_capture['q'].cuda()  # [1, num_heads, head_dim]
    q_batched = q.unsqueeze(1)  # [1, 1, num_heads, head_dim]

    if debug:
        print(f"  Reference for L{layer_id} D{decode_step}:")
        print(f"    q shape: {q_batched.shape}, mean={q_batched.mean().item():.4f}")

    o_acc, lse_acc = None, None

    # Attend to all prefill chunks
    if layer_id in prefill_kv:
        for chunk_data in sorted(prefill_kv[layer_id], key=lambda x: x['chunk_idx']):
            k = chunk_data['k'].cuda().unsqueeze(0)  # [1, seqlen, kv_heads, head_dim]
            v = chunk_data['v'].cuda().unsqueeze(0)

            o, lse = flash_attn_with_lse(q_batched, k, v, softmax_scale=scale, causal=False)

            if debug:
                print(f"    Prefill chunk {chunk_data['chunk_idx']}: o.mean={o.mean().item():.6f}")

            if o_acc is None:
                o_acc, lse_acc = o, lse
            else:
                o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o, lse)

    # Attend to previous decode tokens (including current)
    # In decode, the current token's k,v are stored, and we need to attend to all previous decode tokens
    # For step 0, we just have the current token's k,v
    # For step 1, we have tokens 0 and 1's k,v
    # etc.

    # Collect k,v from all decode steps up to and including current
    decode_kv = []
    for i in range(decode_step + 1):
        if i < len(decode_captures):
            decode_kv.append({
                'k': decode_captures[i]['k'].cuda(),
                'v': decode_captures[i]['v'].cuda(),
            })

    if decode_kv:
        # Stack decode k,v into a single tensor
        decode_k = torch.cat([d['k'] for d in decode_kv], dim=0).unsqueeze(0)  # [1, num_decode, kv_heads, head_dim]
        decode_v = torch.cat([d['v'] for d in decode_kv], dim=0).unsqueeze(0)

        if debug:
            print(f"    Decode tokens: {len(decode_kv)}, k.shape={decode_k.shape}")

        # For decode, we use causal=False since we're attending to all decode tokens
        # (the causal masking was already handled by only including tokens up to current)
        o_decode, lse_decode = flash_attn_with_lse(q_batched, decode_k, decode_v,
                                                    softmax_scale=scale, causal=False)

        if debug:
            print(f"    Decode attention: o.mean={o_decode.mean().item():.6f}")

        if o_acc is None:
            o_acc, lse_acc = o_decode, lse_decode
        else:
            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o_decode, lse_decode)

    if o_acc is None:
        return None

    if debug:
        print(f"    Final: o.mean={o_acc.mean().item():.6f}")

    return o_acc.squeeze(0).squeeze(0).cpu()  # [num_heads, head_dim]


# ============================================================
# Test Runner
# ============================================================

def run_test(verbose=True):
    """Run the hook-based chunked decode correctness test."""
    global captures, prefill_kv
    captures = []
    prefill_kv = {}

    if verbose:
        print("=" * 70)
        print("Test: Hook-Based Chunked Decode Correctness")
        print("=" * 70)
        print(f"Model: {MODEL_PATH}")
        print(f"Input length: {INPUT_LEN} tokens")
        print(f"Decode tokens: {NUM_DECODE_TOKENS}")
        print(f"Block size: {BLOCK_SIZE}")
        print()

    # Initialize LLM with CPU offload
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        enable_cpu_offload=True,
        kvcache_block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
    )

    # Get model info
    num_layers = len(llm.model_runner.model.model.layers)
    head_dim = llm.model_runner.model.model.layers[0].self_attn.attn.head_dim
    scale = head_dim ** -0.5

    if verbose:
        print(f"Num layers: {num_layers}")
        print(f"Head dim: {head_dim}")
        print()

    # Register hooks
    hooks = register_hooks(llm)
    if verbose:
        print(f"Registered {len(hooks)} hooks")

    # Generate random prompt
    seed(42)
    prompt_token_ids = [[randint(0, 10000) for _ in range(INPUT_LEN)]]

    # Run prefill and decode
    if verbose:
        print(f"Running inference with {NUM_DECODE_TOKENS} decode tokens...")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=NUM_DECODE_TOKENS)
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # =========== VERIFICATION: Check CPU cache after prefill ===========
    # Verify that CPU cache data matches captured prefill k,v
    if verbose:
        print("\n--- CPU Cache Verification (After Prefill) ---")
        offload_engine = llm.model_runner.kvcache_manager.offload_engine

        # For each prefill capture, check if CPU cache matches
        for layer_id in [0]:  # Only check layer 0 for brevity
            if layer_id not in prefill_kv:
                continue

            for chunk_data in prefill_kv[layer_id]:
                chunk_idx = chunk_data['chunk_idx']
                captured_k = chunk_data['k']  # [block_size, kv_heads, head_dim]

                # CPU block ID should be chunk_idx (based on allocation order)
                cpu_block_id = chunk_idx
                cpu_k = offload_engine.k_cache_cpu[layer_id, cpu_block_id].cpu()

                diff = (captured_k - cpu_k).abs().max().item()
                print(f"Layer {layer_id}, Chunk {chunk_idx}: captured_k vs cpu_k max_diff={diff:.6f}")
                if diff > 1e-3:
                    print(f"  WARNING: CPU cache doesn't match captured k!")
                    print(f"  captured_k[0,0,:5] = {captured_k[0,0,:5].tolist()}")
                    print(f"  cpu_k[0,0,:5] = {cpu_k[0,0,:5].tolist()}")
        print()

    # Analyze captures
    prefill_count = sum(1 for c in captures if c['is_prefill'])
    decode_count = sum(1 for c in captures if not c['is_prefill'])
    if verbose:
        print(f"\nCaptured {prefill_count} prefill calls, {decode_count} decode calls")

    # Count decode steps per layer
    decode_per_layer = {}
    for c in captures:
        if not c['is_prefill']:
            layer_id = c['layer_id']
            if layer_id not in decode_per_layer:
                decode_per_layer[layer_id] = 0
            decode_per_layer[layer_id] += 1

    if verbose:
        print(f"Decode calls per layer: {decode_per_layer}")
        print()

    # Verify decode correctness
    all_passed = True
    results = []
    first_fail_debug = True

    for c in captures:
        if c['is_prefill']:
            continue  # Skip prefill (already tested in test_chunked_prefill_hook.py)

        layer_id = c['layer_id']
        decode_step = c['decode_step']

        # Only test first decode step for now (simpler reference computation)
        if decode_step > 0:
            continue

        # Compute reference (debug first failure)
        debug_this = (layer_id == 0 and first_fail_debug)
        ref_output = compute_decode_reference(layer_id, decode_step, scale, debug=debug_this)
        if ref_output is None:
            continue

        # Compare
        actual_output = c['output'].squeeze(0)  # Remove seq dim for decode
        if actual_output.dim() == 3:
            actual_output = actual_output.squeeze(0)  # Handle [1, heads, dim] case

        diff = (actual_output - ref_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        tol = 1e-2
        passed = max_diff < tol
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        results.append((layer_id, decode_step, passed, max_diff, mean_diff))

        if verbose:
            print(f"[{status}] Layer {layer_id:2d}, Decode {decode_step}: "
                  f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")

            # Debug first failure
            if not passed and first_fail_debug:
                first_fail_debug = False
                print(f"  Debug: actual_output shape={actual_output.shape}, mean={actual_output.mean().item():.4f}")
                print(f"  Debug: ref_output shape={ref_output.shape}, mean={ref_output.mean().item():.4f}")
                # Find where max diff is
                max_idx = diff.argmax()
                flat_actual = actual_output.flatten()
                flat_ref = ref_output.flatten()
                print(f"  Debug: max_diff at idx={max_idx.item()}, actual={flat_actual[max_idx].item():.4f}, ref={flat_ref[max_idx].item():.4f}")

    print()
    print("=" * 70)

    # Summary
    total_tests = len(results)
    passed_count = sum(1 for r in results if r[2])

    print(f"Results: {passed_count}/{total_tests} tests passed")

    if not all_passed:
        print("\nFailed tests:")
        for layer_id, decode_step, passed, max_diff, mean_diff in results:
            if not passed:
                print(f"  - Layer {layer_id}, Decode {decode_step}: max_diff={max_diff:.6f}")

    print()
    return all_passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    passed = run_test(verbose=True)

    if passed:
        print("test_chunked_decode_hook: PASSED")
    else:
        print("test_chunked_decode_hook: FAILED")
        exit(1)
