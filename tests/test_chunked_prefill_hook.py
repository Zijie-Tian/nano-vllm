"""
Hook-based correctness test for chunked prefill attention.

Uses PyTorch register_forward_hook() to capture real inference I/O,
then compares against reference computation to locate bugs.

This test targets the integration layer (context setup, cpu_block_table management)
which is where the needle test fails despite isolated attention tests passing.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context
from nanovllm.kvcache.chunked_attention import flash_attn_with_lse, merge_attention_outputs
from flash_attn.flash_attn_interface import flash_attn_varlen_func


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
MAX_MODEL_LEN = 32 * 1024
NUM_GPU_BLOCKS = 2
INPUT_LEN = 16 * 1024  # 4K tokens = 4 chunks with 1K block size
BLOCK_SIZE = 1024


# ============================================================
# Global capture storage
# ============================================================

captures = []


# ============================================================
# Hook Functions
# ============================================================

def make_hook(layer_id):
    """Create a forward hook for a specific layer."""
    def hook(module, inputs, output):
        q, k, v = inputs
        ctx = get_context()

        # Only capture prefill phase
        if not ctx.is_prefill:
            return

        chunk_idx = ctx.current_chunk_idx if hasattr(ctx, 'current_chunk_idx') else 0

        capture_entry = {
            'layer_id': layer_id,
            'chunk_idx': chunk_idx,
            'q': q.clone().cpu(),
            'k': k.clone().cpu(),
            'v': v.clone().cpu(),
            'output': output.clone().cpu(),
            'is_chunked_prefill': ctx.is_chunked_prefill,
        }

        # For debugging: also capture CPU cache state for layer 0
        if layer_id == 0 and chunk_idx >= 2:
            kvcache_manager = ctx.kvcache_manager if hasattr(ctx, 'kvcache_manager') else None
            if kvcache_manager is not None and hasattr(kvcache_manager, 'offload_engine'):
                oe = kvcache_manager.offload_engine
                # Get what should have been loaded from CPU
                cpu_k0 = oe.k_cache_cpu[0, 0].clone().cpu()  # Layer 0, CPU block 0
                cpu_k1 = oe.k_cache_cpu[0, 1].clone().cpu()  # Layer 0, CPU block 1
                capture_entry['cpu_k0'] = cpu_k0
                capture_entry['cpu_k1'] = cpu_k1

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

def compute_reference(layer_id, chunk_idx, scale, debug=False):
    """
    Compute reference attention output for a specific layer and chunk.

    Uses the captured k, v from all chunks up to and including chunk_idx.
    """
    # Filter captures for this layer
    layer_captures = [c for c in captures
                      if c['layer_id'] == layer_id and c['chunk_idx'] <= chunk_idx]

    if not layer_captures:
        return None

    # Get current chunk's q
    current_capture = [c for c in layer_captures if c['chunk_idx'] == chunk_idx][0]
    q = current_capture['q'].cuda().unsqueeze(0)  # [1, seqlen, nheads, headdim]

    # Collect all k, v up to current chunk
    kv_list = []
    for c in sorted(layer_captures, key=lambda x: x['chunk_idx']):
        k = c['k'].cuda().unsqueeze(0)  # [1, seqlen, nheads, headdim]
        v = c['v'].cuda().unsqueeze(0)
        kv_list.append((k, v, c['chunk_idx']))

    if debug:
        print(f"  Reference for L{layer_id} C{chunk_idx}:")
        print(f"    q shape: {q.shape}, mean={q.mean().item():.4f}")
        print(f"    kv_list: {len(kv_list)} chunks")
        for i, (k, v, cidx) in enumerate(kv_list):
            print(f"      chunk {cidx}: k.mean={k.mean().item():.4f}, v.mean={v.mean().item():.4f}")

    o_acc, lse_acc = None, None

    # Previous chunks: non-causal attention
    for i in range(len(kv_list) - 1):
        k, v, _ = kv_list[i]
        o, lse = flash_attn_with_lse(q, k, v, softmax_scale=scale, causal=False)
        if o_acc is None:
            o_acc, lse_acc = o, lse
        else:
            o_acc, lse_acc = merge_attention_outputs(o_acc, lse_acc, o, lse)

    # Current chunk: causal attention
    k_cur, v_cur, _ = kv_list[-1]
    o_cur, lse_cur = flash_attn_with_lse(q, k_cur, v_cur, softmax_scale=scale, causal=True)

    if o_acc is None:
        return o_cur.squeeze(0).cpu()

    final_o, _ = merge_attention_outputs(o_acc, lse_acc, o_cur, lse_cur)
    return final_o.squeeze(0).cpu()


def compute_standard_reference(layer_id, chunk_idx, scale, debug=False):
    """
    Compute reference using standard flash attention (single pass with all K, V).

    This simulates what standard (non-chunked) prefill would produce.
    Concatenates all Q, K, V from chunks 0 to chunk_idx and runs a single
    causal attention pass, then extracts the output for the current chunk.
    """
    # Filter captures for this layer
    layer_captures = [c for c in captures
                      if c['layer_id'] == layer_id and c['chunk_idx'] <= chunk_idx]

    if not layer_captures:
        return None

    # Sort by chunk index
    layer_captures = sorted(layer_captures, key=lambda x: x['chunk_idx'])

    # Concatenate all Q, K, V
    all_q = []
    all_k = []
    all_v = []
    chunk_lengths = []

    for c in layer_captures:
        q = c['q'].cuda()  # [seqlen, nheads, headdim]
        k = c['k'].cuda()
        v = c['v'].cuda()
        all_q.append(q)
        all_k.append(k)
        all_v.append(v)
        chunk_lengths.append(q.shape[0])

    # Concatenate along sequence dimension
    full_q = torch.cat(all_q, dim=0)  # [total_seqlen, nheads, headdim]
    full_k = torch.cat(all_k, dim=0)
    full_v = torch.cat(all_v, dim=0)

    total_len = full_q.shape[0]

    if debug:
        print(f"  Standard Reference for L{layer_id} C{chunk_idx}:")
        print(f"    full_q shape: {full_q.shape}, mean={full_q.mean().item():.4f}")
        print(f"    full_k shape: {full_k.shape}, mean={full_k.mean().item():.4f}")
        print(f"    chunk_lengths: {chunk_lengths}")

    # Run standard causal flash attention
    # flash_attn_varlen_func expects: q, k, v with shape [total_seqlen, nheads, headdim]
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

    # Extract output for current chunk only
    start_pos = sum(chunk_lengths[:-1])
    end_pos = sum(chunk_lengths)
    chunk_output = full_o[start_pos:end_pos]

    if debug:
        print(f"    full_o shape: {full_o.shape}")
        print(f"    extracting positions [{start_pos}:{end_pos}]")
        print(f"    chunk_output shape: {chunk_output.shape}, mean={chunk_output.mean().item():.4f}")

    return chunk_output.cpu()


# ============================================================
# Test Runner
# ============================================================

def run_test(verbose=True):
    """Run the hook-based chunked prefill correctness test."""
    global captures
    captures = []

    if verbose:
        print("=" * 70)
        print("Test: Hook-Based Chunked Prefill Correctness")
        print("=" * 70)
        print(f"Model: {MODEL_PATH}")
        print(f"Input length: {INPUT_LEN} tokens")
        print(f"Block size: {BLOCK_SIZE}")
        print(f"Expected chunks: {INPUT_LEN // BLOCK_SIZE}")
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

    # Run prefill only (max_tokens=1)
    if verbose:
        print("Running inference...")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1)
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze captures
    if verbose:
        print(f"\nCaptured {len(captures)} attention calls")

    # Group by layer and chunk
    chunks_per_layer = {}
    for c in captures:
        layer_id = c['layer_id']
        chunk_idx = c['chunk_idx']
        if layer_id not in chunks_per_layer:
            chunks_per_layer[layer_id] = set()
        chunks_per_layer[layer_id].add(chunk_idx)

    if verbose:
        print("Chunks per layer:", {k: sorted(v) for k, v in chunks_per_layer.items()})
        print()

    # First, verify CPU cache data integrity
    if verbose:
        print("\n--- CPU Cache Verification (Layer 0) ---")
        # Get original k from chunk 0 and chunk 1 captures
        chunk0_k = None
        chunk1_k = None
        chunk2_capture = None
        for c in captures:
            if c['layer_id'] == 0:
                if c['chunk_idx'] == 0:
                    chunk0_k = c['k']
                elif c['chunk_idx'] == 1:
                    chunk1_k = c['k']
                elif c['chunk_idx'] == 2:
                    chunk2_capture = c

        if chunk0_k is not None and chunk2_capture is not None and 'cpu_k0' in chunk2_capture:
            cpu_k0 = chunk2_capture['cpu_k0']
            diff_k0 = (chunk0_k - cpu_k0).abs().max().item()
            print(f"Chunk 0 k vs CPU cache block 0: max_diff={diff_k0:.6f}")
            if diff_k0 > 1e-3:
                print(f"  WARNING: CPU cache block 0 differs from original chunk 0 k!")
                print(f"  Original k[0,0,:5] = {chunk0_k[0,0,:5].tolist()}")
                print(f"  CPU k0[0,0,:5] = {cpu_k0[0,0,:5].tolist()}")

        if chunk1_k is not None and chunk2_capture is not None and 'cpu_k1' in chunk2_capture:
            cpu_k1 = chunk2_capture['cpu_k1']
            diff_k1 = (chunk1_k - cpu_k1).abs().max().item()
            print(f"Chunk 1 k vs CPU cache block 1: max_diff={diff_k1:.6f}")
            if diff_k1 > 1e-3:
                print(f"  WARNING: CPU cache block 1 differs from original chunk 1 k!")
                print(f"  Original k[0,0,:5] = {chunk1_k[0,0,:5].tolist()}")
                print(f"  CPU k1[0,0,:5] = {cpu_k1[0,0,:5].tolist()}")

        print()

    # ================================================================
    # Test 1: Verify against merge-based reference (same algorithm)
    # ================================================================
    if verbose:
        print("--- Test 1: Merge-based Reference (verifies merge algorithm) ---")

    all_passed_merge = True
    results_merge = []
    first_fail_debug = True

    for c in captures:
        layer_id = c['layer_id']
        chunk_idx = c['chunk_idx']

        if chunk_idx == 0:
            continue

        debug_this = (chunk_idx >= 2 and layer_id == 0 and first_fail_debug)
        ref_output = compute_reference(layer_id, chunk_idx, scale, debug=debug_this)
        if ref_output is None:
            continue

        actual_output = c['output']
        diff = (actual_output - ref_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        tol = 1e-2
        passed = max_diff < tol
        all_passed_merge = all_passed_merge and passed

        status = "PASS" if passed else "FAIL"
        results_merge.append((layer_id, chunk_idx, passed, max_diff, mean_diff))

        if verbose:
            print(f"[{status}] Layer {layer_id:2d}, Chunk {chunk_idx}: "
                  f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")

            if not passed and first_fail_debug:
                first_fail_debug = False
                print(f"  Debug: actual_output shape={actual_output.shape}, mean={actual_output.mean().item():.4f}")
                print(f"  Debug: ref_output shape={ref_output.shape}, mean={ref_output.mean().item():.4f}")
                max_idx = diff.argmax()
                flat_actual = actual_output.flatten()
                flat_ref = ref_output.flatten()
                print(f"  Debug: max_diff at idx={max_idx.item()}, actual={flat_actual[max_idx].item():.4f}, ref={flat_ref[max_idx].item():.4f}")

    print()

    # ================================================================
    # Test 2: Verify against standard flash attention (single pass)
    # ================================================================
    if verbose:
        print("--- Test 2: Standard FlashAttn Reference (verifies correctness vs non-chunked) ---")

    all_passed_standard = True
    results_standard = []
    first_fail_debug = True

    for c in captures:
        layer_id = c['layer_id']
        chunk_idx = c['chunk_idx']

        if chunk_idx == 0:
            continue

        debug_this = (chunk_idx >= 2 and layer_id == 0 and first_fail_debug)
        std_ref_output = compute_standard_reference(layer_id, chunk_idx, scale, debug=debug_this)
        if std_ref_output is None:
            continue

        actual_output = c['output']
        diff = (actual_output - std_ref_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        tol = 1e-2
        passed = max_diff < tol
        all_passed_standard = all_passed_standard and passed

        status = "PASS" if passed else "FAIL"
        results_standard.append((layer_id, chunk_idx, passed, max_diff, mean_diff))

        if verbose:
            print(f"[{status}] Layer {layer_id:2d}, Chunk {chunk_idx}: "
                  f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")

            if not passed and first_fail_debug:
                first_fail_debug = False
                print(f"  Debug: actual_output shape={actual_output.shape}, mean={actual_output.mean().item():.4f}")
                print(f"  Debug: std_ref_output shape={std_ref_output.shape}, mean={std_ref_output.mean().item():.4f}")
                max_idx = diff.argmax()
                flat_actual = actual_output.flatten()
                flat_ref = std_ref_output.flatten()
                print(f"  Debug: max_diff at idx={max_idx.item()}, actual={flat_actual[max_idx].item():.4f}, ref={flat_ref[max_idx].item():.4f}")

    print()
    print("=" * 70)

    # Summary
    total_merge = len(results_merge)
    passed_merge = sum(1 for r in results_merge if r[2])
    total_standard = len(results_standard)
    passed_standard = sum(1 for r in results_standard if r[2])

    print(f"Merge-based reference:    {passed_merge}/{total_merge} tests passed")
    print(f"Standard FlashAttn ref:   {passed_standard}/{total_standard} tests passed")

    all_passed = all_passed_merge and all_passed_standard

    if not all_passed_merge:
        print("\nFailed merge-based tests:")
        for layer_id, chunk_idx, passed, max_diff, mean_diff in results_merge:
            if not passed:
                print(f"  - Layer {layer_id}, Chunk {chunk_idx}: max_diff={max_diff:.6f}")

    if not all_passed_standard:
        print("\nFailed standard FlashAttn tests:")
        for layer_id, chunk_idx, passed, max_diff, mean_diff in results_standard:
            if not passed:
                print(f"  - Layer {layer_id}, Chunk {chunk_idx}: max_diff={max_diff:.6f}")

    print()
    return all_passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    passed = run_test(verbose=True)

    if passed:
        print("test_chunked_prefill_hook: PASSED")
    else:
        print("test_chunked_prefill_hook: FAILED")
        exit(1)
