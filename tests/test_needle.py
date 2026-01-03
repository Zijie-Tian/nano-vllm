"""
Needle-in-a-haystack test for LLM.

Tests: Long context retrieval capability with configurable sequence length.

NOTE: CPU offload mode has a known bug that causes incorrect outputs for
sequences longer than ~200 tokens. Use --no-offload for correctness testing.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

import argparse
from nanovllm import LLM, SamplingParams
from utils import generate_needle_prompt, check_needle_answer


# ============================================================
# Main Test
# ============================================================

def run_needle_test(
    model_path: str,
    max_model_len: int,
    input_len: int,
    num_gpu_blocks: int = 4,
    needle_position: float = 0.5,
    needle_value: str = "7492",
    max_new_tokens: int = 32,
    enable_cpu_offload: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run a needle-in-haystack test.

    Args:
        model_path: Path to model
        max_model_len: Maximum model context length
        input_len: Target input sequence length
        num_gpu_blocks: Number of GPU blocks for offload
        needle_position: Where to place needle (0.0-1.0)
        needle_value: The secret value to find
        max_new_tokens: Maximum tokens to generate
        enable_cpu_offload: Enable CPU offload mode
        verbose: Print detailed output

    Returns:
        True if test passed, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Needle-in-Haystack Test")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Max model len: {max_model_len}")
        print(f"Input length: {input_len}")
        print(f"Needle position: {needle_position:.0%}")
        print(f"Needle value: {needle_value}")
        print(f"CPU offload: {enable_cpu_offload}")
        print(f"{'='*60}\n")

    # 1. Initialize LLM
    llm_kwargs = {
        "enforce_eager": True,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "enable_cpu_offload": enable_cpu_offload,
    }
    if enable_cpu_offload:
        llm_kwargs["num_gpu_blocks"] = num_gpu_blocks

    llm = LLM(model_path, **llm_kwargs)

    # 2. Generate needle prompt
    prompt, expected = generate_needle_prompt(
        tokenizer=llm.tokenizer,
        target_length=input_len,
        needle_position=needle_position,
        needle_value=needle_value,
    )

    # 3. Generate output
    sampling_params = SamplingParams(
        temperature=0.6,  # Moderate temperature
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=True)

    # 4. Check result
    output_text = outputs[0]["text"]
    output_token_ids = outputs[0]["token_ids"]
    passed = check_needle_answer(output_text, expected)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Result")
        print(f"{'='*60}")
        print(f"Expected: {expected}")
        print(f"Output tokens ({len(output_token_ids)}): {output_token_ids[:20]}")
        print(f"Output: {output_text[:200]}...")
        print(f"Status: {'PASSED' if passed else 'FAILED'}")
        print(f"{'='*60}\n")

    return passed


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Needle-in-haystack test for long context LLM")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/"),
        help="Path to model"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32 * 1024,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=8 * 1024,
        help="Target input sequence length"
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=2,
        help="Number of GPU blocks for CPU offload"
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
        "--enable-offload",
        action="store_true",
        help="Enable CPU offload (has known bug for long sequences)"
    )
    args = parser.parse_args()

    passed = run_needle_test(
        model_path=args.model,
        max_model_len=args.max_model_len,
        input_len=args.input_len,
        num_gpu_blocks=args.num_gpu_blocks,
        needle_position=args.needle_position,
        needle_value=args.needle_value,
        max_new_tokens=args.max_new_tokens,
        enable_cpu_offload=args.enable_offload,
        verbose=True,
    )

    if passed:
        print("test_needle: PASSED")
    else:
        print("test_needle: FAILED")
        exit(1)
