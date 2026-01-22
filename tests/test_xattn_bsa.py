"""
Test XAttention + BSA with RULER benchmark data.

Tests XAttention sparse attention correctness using RULER NIAH task.

Attention methods:
  - Prefill: XAttention + BSA (sparse) or FlashAttention (dense)
  - Decode:  FlashAttention (always, since q_len=1)

Usage (in compass conda env with BSA available):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH \
        python tests/test_xattn_bsa.py --model ~/models/Llama-3.1-8B-Instruct

    # Test with XAttention + BSA for prefill (default)
    python tests/test_xattn_bsa.py --prefill-method xattn

    # Test with FlashAttention for prefill (baseline)
    python tests/test_xattn_bsa.py --prefill-method flash

    # Test specific sample(s)
    python tests/test_xattn_bsa.py --sample-id 0
    python tests/test_xattn_bsa.py --sample-ids 0,1,2

Note: Compatible with transformers 4.53+ (handles both old `past_key_value`
      and new `past_key_values` API).
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from nanovllm.ops.xattn import xattn_estimate
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


# ============================================================
# XAttention + BSA Functions
# ============================================================

def expand_kv_for_gqa(key_states, value_states, num_heads):
    """Expand KV for Grouped Query Attention."""
    num_kv_heads = key_states.shape[1]
    if num_heads == num_kv_heads:
        return key_states, value_states
    num_groups = num_heads // num_kv_heads
    return key_states.repeat_interleave(num_groups, dim=1), value_states.repeat_interleave(num_groups, dim=1)


def flash_attention_forward(query_states, key_states, value_states, is_causal=True):
    """Standard FlashAttention."""
    from flash_attn import flash_attn_func
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    return flash_attn_func(q, k, v, causal=is_causal).transpose(1, 2)


def xattn_bsa_forward(query_states, key_states, value_states, threshold=0.9):
    """XAttention + BSA sparse attention."""
    from block_sparse_attn import block_sparse_attn_func

    batch_size, num_heads, q_len, head_dim = query_states.shape
    k_len = key_states.shape[2]

    _, mask = xattn_estimate(
        query_states, key_states,
        chunk_size=16384, block_size=128, threshold=threshold,
        use_triton=True, causal=True,
    )

    q_block_num = (q_len + 127) // 128
    k_block_num = (k_len + 127) // 128

    q = query_states.transpose(1, 2).reshape(q_len, num_heads, head_dim)
    k = key_states.transpose(1, 2).reshape(k_len, num_heads, head_dim)
    v = value_states.transpose(1, 2).reshape(k_len, num_heads, head_dim)
    
    __import__('pdb').set_trace()

    output = block_sparse_attn_func(
        q, k, v,
        torch.tensor([0, q_len], dtype=torch.int32, device=q.device),
        torch.tensor([0, k_len], dtype=torch.int32, device=k.device),
        torch.ones(num_heads, dtype=torch.int32, device=q.device),
        None,
        mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len, k_len,
        p_dropout=0.0, deterministic=True, is_causal=True,
    )
    return output.reshape(batch_size, q_len, num_heads, head_dim).transpose(1, 2)


DEBUG = False  # Set to True to enable debugging

def create_patched_forward(prefill_method="xattn", threshold=0.9):
    """Create patched forward with configurable prefill method.

    Args:
        prefill_method: "xattn" for XAttention + BSA (sparse), "flash" for FlashAttention (dense)
        threshold: XAttention threshold for block selection (only used when prefill_method="xattn")

    Note:
        - Prefill (q_len > 1): Uses specified prefill_method
        - Decode (q_len = 1): Always uses FlashAttention (no sparse needed for single query)
    """
    call_count = [0]  # Mutable to track calls across layers

    def patched_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_value=None,   # Old API (transformers < 4.57)
        past_key_values=None,  # New API (transformers >= 4.57)
        cache_position=None,
        **kwargs
    ):
        # Handle both old and new transformers API
        kv_cache = past_key_values if past_key_values is not None else past_key_value

        bsz, q_len, _ = hidden_states.size()
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim

        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary position embedding
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if kv_cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Expand KV for GQA
        key_states_exp, value_states_exp = expand_kv_for_gqa(key_states, value_states, num_heads)

        # Debug output
        if DEBUG and self.layer_idx == 0:
            call_count[0] += 1
            if call_count[0] <= 5:
                phase = "prefill" if q_len > 1 else "decode"
                print(f"\n[DEBUG] Layer {self.layer_idx}, call {call_count[0]} ({phase}): q_len={q_len}, k_len={key_states_exp.shape[2]}")
                print(f"  kv_cache is None: {kv_cache is None}")

        # Choose attention method:
        # - Prefill (q_len > 1): Use prefill_method (xattn or flash)
        # - Decode (q_len = 1): Always use FlashAttention
        is_prefill = q_len > 1

        if is_prefill and prefill_method == "xattn":
            # Prefill with XAttention + BSA (sparse)
            attn_output = xattn_bsa_forward(query_states, key_states_exp, value_states_exp, threshold)
        else:
            # Prefill with FlashAttention (dense) OR Decode (always FlashAttention)
            # Note: For decode (q_len=1), causal=False since single query attends to all KV
            attn_output = flash_attention_forward(query_states, key_states_exp, value_states_exp, is_causal=is_prefill)

        attn_output = self.o_proj(attn_output.transpose(1, 2).reshape(bsz, q_len, -1))
        return attn_output, None

    return patched_forward


# ============================================================
# Data & Evaluation
# ============================================================

def load_samples(filepath, indices=None):
    """Load samples from JSONL file."""
    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if indices is None or i in indices:
                sample = json.loads(line)
                sample["_idx"] = i
                samples.append(sample)
    return samples


def string_match_all(output_text, expected_list):
    """RULER metric: fraction of expected values found in output."""
    output_lower = output_text.lower().replace('\n', ' ')
    if not expected_list:
        return 1.0
    return sum(1.0 if exp.strip().lower() in output_lower else 0.0 for exp in expected_list) / len(expected_list)


# ============================================================
# Test
# ============================================================

def test_with_ruler_data(model_path, data_file, sample_ids, prefill_method="xattn", threshold=0.9, max_new_tokens=50):
    """Test attention methods using RULER data.

    Args:
        prefill_method: "xattn" for XAttention + BSA, "flash" for FlashAttention
    """
    prefill_desc = "XAttention + BSA (sparse)" if prefill_method == "xattn" else "FlashAttention (dense)"

    print("=" * 60)
    print("RULER NIAH Attention Test")
    print("=" * 60)
    print(f"Data: {data_file}")
    print(f"Samples: {sample_ids}")
    print(f"Prefill method: {prefill_desc}")
    print(f"Decode method:  FlashAttention (always)")
    if prefill_method == "xattn":
        print(f"XAttention threshold: {threshold}")

    samples = load_samples(Path(data_file), set(sample_ids) if sample_ids else None)
    if not samples:
        print("No samples found!")
        return False
    print(f"Loaded {len(samples)} samples")

    # Load model
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",  # Will be patched
    )
    model.eval()

    # Patch all layers
    print(f"Patching attention layers...")
    print(f"  - Prefill: {prefill_desc}")
    print(f"  - Decode:  FlashAttention")
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = idx  # Ensure layer_idx is set
        layer.self_attn.forward = create_patched_forward(prefill_method, threshold).__get__(
            layer.self_attn, type(layer.self_attn)
        )

    total_score = 0.0
    results = []

    for sample in samples:
        idx = sample["_idx"]
        prompt = sample["input"]
        expected = sample["outputs"]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        num_tokens = inputs["input_ids"].shape[1]
        print(f"\n--- Sample {idx} ({num_tokens} tokens) ---")
        print(f"Expected: {expected}")

        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output[0][num_tokens:], skip_special_tokens=True)
        score = string_match_all(output_text, expected)
        total_score += score

        status = "✓ PASS" if score >= 0.5 else "✗ FAIL"
        print(f"Output: '{output_text[:100]}...'")
        print(f"Result: {status} (score={score:.2f})")
        results.append({"idx": idx, "score": score, "passed": score >= 0.5})

    avg_score = total_score / len(samples)
    passed = sum(1 for r in results if r["passed"])

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(samples)} passed, avg_score={avg_score:.3f}")
    print(f"{'='*60}")

    return avg_score >= 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Test XAttention + BSA vs FlashAttention for prefill using RULER NIAH benchmark"
    )
    parser.add_argument("--model", default="~/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--data-file", default="tests/data/ruler_32k/niah_single_1/validation.jsonl")
    parser.add_argument("--sample-id", type=int, default=None, help="Test single sample by index")
    parser.add_argument("--sample-ids", type=str, default="", help="Test multiple samples (comma-separated)")
    parser.add_argument("--prefill-method", choices=["xattn", "flash"], default="xattn",
                        help="Prefill attention method: xattn (XAttention+BSA sparse) or flash (FlashAttention dense)")
    parser.add_argument("--threshold", type=float, default=0.9, help="XAttention threshold (only for --prefill-method xattn)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    # Keep old option for backwards compatibility
    parser.add_argument("--no-xattn", action="store_true", help="[Deprecated] Use --prefill-method flash instead")
    args = parser.parse_args()

    model_path = args.model.replace("~", "/home/zijie")

    # Handle deprecated --no-xattn option
    prefill_method = args.prefill_method
    if args.no_xattn:
        prefill_method = "flash"
        print("Warning: --no-xattn is deprecated, use --prefill-method flash instead")

    if args.sample_id is not None:
        sample_ids = [args.sample_id]
    elif args.sample_ids:
        sample_ids = [int(x) for x in args.sample_ids.split(",")]
    else:
        sample_ids = [0]

    # Check BSA availability if using xattn
    if prefill_method == "xattn":
        try:
            from block_sparse_attn import block_sparse_attn_func
            print("✓ BSA (Block Sparse Attention) available")
        except ImportError:
            print("✗ BSA not found. Install block_sparse_attn or use --prefill-method flash")
            sys.exit(1)

    if test_with_ruler_data(model_path, args.data_file, sample_ids, prefill_method, args.threshold, args.max_new_tokens):
        print("\ntest_xattn_bsa: PASSED")
    else:
        print("\ntest_xattn_bsa: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
