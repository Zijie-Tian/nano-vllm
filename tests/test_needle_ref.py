"""
Needle-in-a-haystack reference test using pure torch + transformers.

This is a reference implementation for comparison with nanovllm.
Uses standard HuggingFace inference (no custom KV cache, no offload).
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM
from utils import generate_needle_prompt, check_needle_answer


# ============================================================
# Main Test
# ============================================================

def run_needle_test(
    model_path: str,
    input_len: int,
    needle_position: float = 0.5,
    needle_value: str = "7492",
    max_new_tokens: int = 32,
    dtype: str = "auto",
    verbose: bool = True,
) -> bool:
    """
    Run a needle-in-haystack test using standard transformers inference.

    Args:
        model_path: Path to model
        input_len: Target input sequence length
        needle_position: Where to place needle (0.0-1.0)
        needle_value: The secret value to find
        max_new_tokens: Maximum tokens to generate
        dtype: Model dtype ("auto", "float16", "bfloat16")
        verbose: Print detailed output

    Returns:
        True if test passed, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Needle-in-Haystack Reference Test (torch + transformers)")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Input length: {input_len}")
        print(f"Needle position: {needle_position:.0%}")
        print(f"Needle value: {needle_value}")
        print(f"Dtype: {dtype}")
        print(f"{'='*60}\n")

    # 1. Load tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Generate needle prompt
    print("[2/4] Generating needle prompt...")
    prompt, expected = generate_needle_prompt(
        tokenizer=tokenizer,
        target_length=input_len,
        needle_position=needle_position,
        needle_value=needle_value,
    )

    # 3. Load model
    print("[3/4] Loading model...")
    torch_dtype = {
        "auto": torch.float16,  # default to float16 for custom model
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float16)

    model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 4. Generate output
    print("[4/4] Running inference...")
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"  Input shape: {input_ids.shape}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_token_ids = output_ids[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)

    # 5. Check result
    passed = check_needle_answer(output_text, expected)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Result")
        print(f"{'='*60}")
        print(f"Expected: {expected}")
        print(f"Output tokens ({len(new_token_ids)}): {new_token_ids[:20].tolist()}")
        print(f"Output: {output_text[:200]}...")
        print(f"Status: {'PASSED' if passed else 'FAILED'}")
        print(f"{'='*60}\n")

    return passed


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Needle-in-haystack reference test (torch + transformers)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/"),
        help="Path to model"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=8 * 1024,
        help="Target input sequence length"
    )
    parser.add_argument(
        "--needle-position",
        type=float,
        default=0.5,
        help="Needle position (0.0=start, 0.5=middle, 1.0=end)"
    )
    parser.add_argument(
        "--needle-value",
        type=str,
        default="7492",
        help="The secret value to hide"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Model dtype"
    )
    args = parser.parse_args()

    passed = run_needle_test(
        model_path=args.model,
        input_len=args.input_len,
        needle_position=args.needle_position,
        needle_value=args.needle_value,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        verbose=True,
    )

    if passed:
        print("test_needle_ref: PASSED")
    else:
        print("test_needle_ref: FAILED")
        exit(1)
