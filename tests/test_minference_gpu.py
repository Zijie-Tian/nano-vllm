"""
Needle-in-haystack test with MInference sparse attention.

Tests: MInference sparse prefill on GPU-only path (no CPU offload).
This validates that MInference's vertical + slash sparse pattern can
correctly retrieve information from long context.
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import argparse
from nanovllm import LLM, SamplingParams
from nanovllm.config import SparsePolicyType
from utils import generate_needle_prompt, check_needle_answer


def run_minference_test(
    model_path: str,
    max_model_len: int = 16384,
    input_len: int = 8192,
    needle_position: float = 0.5,
    needle_value: str = "7492",
    adaptive_budget: float = 0.3,
    max_new_tokens: int = 32,
    verbose: bool = True,
) -> bool:
    """
    Run needle test with MInference sparse prefill attention.

    Args:
        model_path: Path to model
        max_model_len: Maximum model context length
        input_len: Target input sequence length
        needle_position: Where to place needle (0.0-1.0)
        needle_value: The secret value to find
        adaptive_budget: MInference budget as fraction of seq_len
        max_new_tokens: Maximum tokens to generate
        verbose: Print detailed output

    Returns:
        True if test passed, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"MInference Sparse Prefill Test (GPU-only)")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Max model len: {max_model_len}")
        print(f"Input length: {input_len}")
        print(f"Needle position: {needle_position:.0%}")
        print(f"Needle value: {needle_value}")
        print(f"Adaptive budget: {adaptive_budget}")
        print(f"{'='*60}\n")

    # Initialize LLM with MInference sparse attention
    llm = LLM(
        model_path,
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        enable_cpu_offload=False,  # GPU-only
        sparse_policy=SparsePolicyType.MINFERENCE,
        minference_adaptive_budget=adaptive_budget,
    )

    # Generate needle prompt
    prompt, expected = generate_needle_prompt(
        tokenizer=llm.tokenizer,
        target_length=input_len,
        needle_position=needle_position,
        needle_value=needle_value,
    )

    # Generate output
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=True)

    # Check result
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Needle-in-haystack test with MInference sparse prefill"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/"),
        help="Path to model"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16 * 1024,
        help="Maximum model context length"
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
        "--adaptive-budget",
        type=float,
        default=0.3,
        help="MInference adaptive budget (fraction of seq_len)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate"
    )
    args = parser.parse_args()

    passed = run_minference_test(
        model_path=args.model,
        max_model_len=args.max_model_len,
        input_len=args.input_len,
        needle_position=args.needle_position,
        needle_value=args.needle_value,
        adaptive_budget=args.adaptive_budget,
        max_new_tokens=args.max_new_tokens,
        verbose=True,
    )

    if passed:
        print("test_minference_gpu: PASSED")
    else:
        print("test_minference_gpu: FAILED")
        exit(1)
