"""
Test utilities for nano-vllm.
"""

import re
from typing import Tuple


# ============================================================
# Needle-in-Haystack Test Utilities
# ============================================================

# Haystack filler paragraphs (various topics to create realistic context)
HAYSTACK_PARAGRAPHS = [
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


def generate_needle_prompt(
    tokenizer,
    target_length: int,
    needle_position: float = 0.5,
    needle_value: str = "7492",
    use_chat_template: bool = True,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Generate a needle-in-haystack prompt of exactly target_length tokens.

    Args:
        tokenizer: HuggingFace tokenizer for length estimation
        target_length: Target total sequence length in tokens
        needle_position: Where to place needle (0.0=start, 0.5=middle, 1.0=end)
        needle_value: The secret value to hide in the haystack
        use_chat_template: Whether to use chat template for instruct models
        verbose: Whether to print generation info

    Returns:
        (prompt, expected_answer): The full prompt and the expected needle value
    """
    # The needle sentence
    needle = f"The secret number you need to remember is {needle_value}. This is very important. "

    # Question text
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        question_text = "/no_think Answer only with the secret number mentioned above, nothing else:"
    else:
        question_text = "\n\nQuestion: What is the secret number mentioned in the text above?\nAnswer: The secret number is"

    def build_prompt(haystack_parts, needle_idx):
        """Build full prompt from haystack parts with needle inserted."""
        parts = haystack_parts.copy()
        parts.insert(needle_idx, needle)
        full_text = "".join(parts)

        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": f"{full_text}\n\n{question_text}"}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return full_text + question_text

    def count_tokens(prompt):
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    def get_needle_idx(parts):
        idx = int(len(parts) * needle_position)
        return max(0, min(idx, len(parts)))

    # Phase 1: Build haystack with full paragraphs until we exceed target
    haystack_parts = []
    para_idx = 0

    while True:
        para = HAYSTACK_PARAGRAPHS[para_idx % len(HAYSTACK_PARAGRAPHS)]
        test_parts = haystack_parts + [para]
        prompt = build_prompt(test_parts, get_needle_idx(test_parts))

        if count_tokens(prompt) > target_length:
            break

        haystack_parts.append(para)
        para_idx += 1

        if para_idx > 10000:  # Safety limit
            break

    # Phase 2: Fine-tune by adding words from next paragraph
    next_para = HAYSTACK_PARAGRAPHS[para_idx % len(HAYSTACK_PARAGRAPHS)]
    words = next_para.split()

    best_parts = haystack_parts.copy()
    best_diff = abs(target_length - count_tokens(build_prompt(haystack_parts, get_needle_idx(haystack_parts))))

    for i in range(1, len(words) + 1):
        partial = " ".join(words[:i]) + " "
        test_parts = haystack_parts + [partial]
        prompt = build_prompt(test_parts, get_needle_idx(test_parts))
        token_count = count_tokens(prompt)
        diff = abs(target_length - token_count)

        if diff < best_diff:
            best_diff = diff
            best_parts = test_parts.copy()

        if token_count >= target_length:
            break

    haystack_parts = best_parts

    # Final build
    needle_idx = get_needle_idx(haystack_parts)
    prompt = build_prompt(haystack_parts, needle_idx)

    actual_tokens = count_tokens(prompt)
    if verbose:
        print(f"[NeedleTest] Target: {target_length}, Actual: {actual_tokens} tokens (diff={actual_tokens - target_length})")

    return prompt, needle_value


def check_needle_answer(output_text: str, expected: str) -> bool:
    """Check if the model output contains the expected needle value."""
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


def generate_random_token_ids(
    length: int,
    vocab_size: int = 10000,
    seed: int = 42,
) -> list:
    """
    Generate random token IDs for testing.

    Args:
        length: Number of tokens to generate
        vocab_size: Maximum token ID (exclusive)
        seed: Random seed for reproducibility

    Returns:
        List of random token IDs
    """
    from random import randint, seed as set_seed
    set_seed(seed)
    return [randint(0, vocab_size - 1) for _ in range(length)]
