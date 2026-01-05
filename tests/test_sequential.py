"""
Sequential inference test for LLM.

Tests: After completing one prompt, the system can correctly handle
a second prompt with a clean state (first prompt's KV cache deallocated).
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import argparse
from nanovllm import LLM, SamplingParams
from utils import generate_needle_prompt, check_needle_answer


def run_sequential_test(
    model_path: str,
    max_model_len: int,
    input_len: int,
    num_gpu_blocks: int = 4,
    block_size: int = 1024,
    enable_cpu_offload: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run sequential inference test with two different prompts.

    Each prompt has a different needle value. Both must be retrieved correctly.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sequential Inference Test")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Max model len: {max_model_len}")
        print(f"Input length: {input_len}")
        print(f"Block size: {block_size}")
        print(f"CPU offload: {enable_cpu_offload}")
        print(f"{'='*60}\n")

    # Initialize LLM once
    llm_kwargs = {
        "enforce_eager": True,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "enable_cpu_offload": enable_cpu_offload,
        "kvcache_block_size": block_size,
    }
    if enable_cpu_offload:
        llm_kwargs["num_gpu_blocks"] = num_gpu_blocks

    llm = LLM(model_path, **llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=32,
    )

    # ============================================================
    # Test 1: First prompt with needle value "1234"
    # ============================================================
    needle_value_1 = "1234"
    if verbose:
        print(f"\n[Test 1] Generating prompt with needle value: {needle_value_1}")

    prompt_1, expected_1 = generate_needle_prompt(
        tokenizer=llm.tokenizer,
        target_length=input_len,
        needle_position=0.5,
        needle_value=needle_value_1,
    )

    outputs_1 = llm.generate([prompt_1], sampling_params, use_tqdm=True)
    output_text_1 = outputs_1[0]["text"]
    passed_1 = check_needle_answer(output_text_1, expected_1)

    if verbose:
        print(f"  Expected: {expected_1}")
        print(f"  Output: {output_text_1[:100]}...")
        print(f"  Status: {'PASSED' if passed_1 else 'FAILED'}")

    # ============================================================
    # Test 2: Second prompt with needle value "5678"
    # ============================================================
    needle_value_2 = "5678"
    if verbose:
        print(f"\n[Test 2] Generating prompt with needle value: {needle_value_2}")

    prompt_2, expected_2 = generate_needle_prompt(
        tokenizer=llm.tokenizer,
        target_length=input_len,
        needle_position=0.5,
        needle_value=needle_value_2,
    )

    outputs_2 = llm.generate([prompt_2], sampling_params, use_tqdm=True)
    output_text_2 = outputs_2[0]["text"]
    passed_2 = check_needle_answer(output_text_2, expected_2)

    if verbose:
        print(f"  Expected: {expected_2}")
        print(f"  Output: {output_text_2[:100]}...")
        print(f"  Status: {'PASSED' if passed_2 else 'FAILED'}")

    # ============================================================
    # Test 3: Third prompt - repeat first needle to ensure no cross-contamination
    # ============================================================
    needle_value_3 = "9999"
    if verbose:
        print(f"\n[Test 3] Generating prompt with needle value: {needle_value_3}")

    prompt_3, expected_3 = generate_needle_prompt(
        tokenizer=llm.tokenizer,
        target_length=input_len,
        needle_position=0.5,
        needle_value=needle_value_3,
    )

    outputs_3 = llm.generate([prompt_3], sampling_params, use_tqdm=True)
    output_text_3 = outputs_3[0]["text"]
    passed_3 = check_needle_answer(output_text_3, expected_3)

    if verbose:
        print(f"  Expected: {expected_3}")
        print(f"  Output: {output_text_3[:100]}...")
        print(f"  Status: {'PASSED' if passed_3 else 'FAILED'}")

    # ============================================================
    # Summary
    # ============================================================
    all_passed = passed_1 and passed_2 and passed_3

    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Test 1 (needle={needle_value_1}): {'PASSED' if passed_1 else 'FAILED'}")
        print(f"Test 2 (needle={needle_value_2}): {'PASSED' if passed_2 else 'FAILED'}")
        print(f"Test 3 (needle={needle_value_3}): {'PASSED' if passed_3 else 'FAILED'}")
        print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
        print(f"{'='*60}\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential inference test")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/"),
        help="Path to model"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=36 * 1024,
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
        "--block-size",
        type=int,
        default=1024,
        help="KV cache block size"
    )
    parser.add_argument(
        "--enable-offload",
        action="store_true",
        help="Enable CPU offload"
    )
    args = parser.parse_args()

    passed = run_sequential_test(
        model_path=args.model,
        max_model_len=args.max_model_len,
        input_len=args.input_len,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        enable_cpu_offload=args.enable_offload,
        verbose=True,
    )

    if passed:
        print("test_sequential: PASSED")
    else:
        print("test_sequential: FAILED")
        exit(1)
