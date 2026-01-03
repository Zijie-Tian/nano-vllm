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
    Generate a needle-in-haystack prompt of approximately target_length tokens.

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
        para = HAYSTACK_PARAGRAPHS[para_idx % len(HAYSTACK_PARAGRAPHS)]
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
    if verbose:
        print(f"[NeedleTest] Target: {target_length} tokens, Actual: {actual_tokens} tokens")
        print(f"[NeedleTest] Needle position: {needle_position:.0%} ({needle_idx}/{len(haystack_parts)-1} paragraphs)")
        print(f"[NeedleTest] Using chat template: {use_chat_template and hasattr(tokenizer, 'apply_chat_template')}")

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
