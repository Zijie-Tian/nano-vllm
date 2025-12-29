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


# ============================================================
# Needle Test Generator
# ============================================================

def generate_needle_prompt(
    tokenizer,
    target_length: int,
    needle_position: float = 0.5,
    needle_value: str = "7492",
    use_chat_template: bool = True,
) -> tuple[str, str]:
    """
    Generate a needle-in-haystack prompt of approximately target_length tokens.

    Args:
        tokenizer: HuggingFace tokenizer for length estimation
        target_length: Target total sequence length in tokens
        needle_position: Where to place needle (0.0=start, 0.5=middle, 1.0=end)
        needle_value: The secret value to hide in the haystack
        use_chat_template: Whether to use chat template for instruct models

    Returns:
        (prompt, expected_answer): The full prompt and the expected needle value
    """
    # Haystack filler paragraphs (various topics to create realistic context)
    haystack_paragraphs = [
        "The weather today is quite pleasant with clear skies and moderate temperatures. "
        "Many people are enjoying outdoor activities in the park. "
        "Birds are singing in the trees and children are playing on the swings. ",

        "In the world of technology, new innovations continue to emerge every day. "
        "Researchers are working on advanced algorithms and computing systems. "
        "The future of artificial intelligence looks promising with many breakthroughs. ",

        "The history of human civilization spans thousands of years. "
        "Ancient cultures developed writing, mathematics, and astronomy. "
        "Trade routes connected distant lands and facilitated cultural exchange. ",

        "Modern cooking combines traditional techniques with new ingredients. "
        "Chefs around the world experiment with flavors and presentations. "
        "Food brings people together and creates memorable experiences. ",

        "The ocean covers more than seventy percent of Earth's surface. "
        "Marine ecosystems support an incredible diversity of life forms. "
        "Scientists continue to discover new species in the deep sea. ",

        "Music has been a part of human culture since prehistoric times. "
        "Different genres evolved across various regions and time periods. "
        "Today, people can access millions of songs through digital platforms. ",

        "Space exploration has revealed many secrets about our universe. "
        "Telescopes can observe galaxies billions of light years away. "
        "Future missions aim to establish human presence on other planets. ",

        "The study of languages reveals patterns in human cognition. "
        "Linguists analyze grammar, semantics, and phonetics across cultures. "
        "Language continues to evolve with new words and expressions. ",
    ]

    # The needle sentence
    needle = f"The secret number you need to remember is {needle_value}. This is very important. "

    # Question at the end
    question = "\n\nQuestion: What is the secret number mentioned in the text above?\nAnswer: The secret number is"

    # Estimate tokens for fixed parts
    needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
    question_text = "What is the secret number mentioned in the text above? Answer with just the number."
    question_tokens = len(tokenizer.encode(question_text, add_special_tokens=False))
    # Buffer for chat template, special tokens, etc.
    overhead_tokens = 100 if use_chat_template else 50

    # Available tokens for haystack
    haystack_target_tokens = target_length - needle_tokens - question_tokens - overhead_tokens
    if haystack_target_tokens < 100:
        raise ValueError(f"target_length {target_length} is too short for needle test")

    # Build haystack by repeating paragraphs
    haystack_parts = []
    current_tokens = 0
    para_idx = 0

    while current_tokens < haystack_target_tokens:
        para = haystack_paragraphs[para_idx % len(haystack_paragraphs)]
        para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
        if current_tokens + para_tokens > haystack_target_tokens:
            break
        haystack_parts.append(para)
        current_tokens += para_tokens
        para_idx += 1

    # Calculate needle insertion point
    needle_idx = int(len(haystack_parts) * needle_position)
    needle_idx = max(0, min(needle_idx, len(haystack_parts)))

    # Insert needle
    haystack_parts.insert(needle_idx, needle)

    # Assemble prompt
    full_text = "".join(haystack_parts)

    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # Use chat template for instruct models
        # For Qwen3, add /no_think to disable thinking mode
        question_text = "/no_think Answer only with the secret number mentioned above, nothing else:"
        messages = [
            {"role": "user", "content": f"{full_text}\n\n{question_text}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Raw text format for base models
        question = "\n\nQuestion: What is the secret number mentioned in the text above?\nAnswer: The secret number is"
        prompt = full_text + question

    # Verify length
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"[NeedleTest] Target: {target_length} tokens, Actual: {actual_tokens} tokens")
    print(f"[NeedleTest] Needle position: {needle_position:.0%} ({needle_idx}/{len(haystack_parts)-1} paragraphs)")
    print(f"[NeedleTest] Using chat template: {use_chat_template and hasattr(tokenizer, 'apply_chat_template')}")

    return prompt, needle_value


def check_needle_answer(output_text: str, expected: str) -> bool:
    """Check if the model output contains the expected needle value."""
    import re
    # Clean output - remove special tokens and whitespace
    output_clean = output_text.replace('<|im_end|>', '').replace('\r', ' ').replace('\n', ' ')
    output_clean = ' '.join(output_clean.split()).lower()
    expected_clean = expected.strip().lower()

    # Check if expected value appears in output
    # Also try to find it as a standalone number
    if expected_clean in output_clean:
        return True

    # Try to extract numbers and check if expected is among them
    numbers = re.findall(r'\d+', output_clean)
    return expected_clean in numbers


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
