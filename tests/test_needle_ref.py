"""
Needle-in-a-haystack reference test using pure torch + transformers.

This is a reference implementation for comparison with nanovllm.
Uses standard HuggingFace inference (no custom KV cache, no offload).
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 4. Generate output
    print("[4/4] Running inference...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
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
