"""
Test chunked attention with small num_gpu_blocks to trigger CPU offload.

For 8K tokens with block_size=256:
- Total blocks needed: 8192 / 256 = 32 blocks
- With num_gpu_blocks=10, 22 blocks go to CPU -> triggers chunked attention
"""
import os
import sys

# Enable debug logging before importing nanovllm
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

from nanovllm import LLM, SamplingParams


def create_long_context_prompt(target_tokens: int) -> str:
    """
    Create a meaningful long context prompt with a question at the end.
    The answer depends on information scattered throughout the context.
    """
    # Key facts to embed in the context
    facts = [
        "The capital of France is Paris.",
        "The Eiffel Tower was built in 1889.",
        "Python was created by Guido van Rossum.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "Mount Everest is 8,848 meters tall.",
    ]

    # Padding text to reach target length
    padding_paragraph = """
This is additional context information that helps extend the length of the prompt.
Machine learning has revolutionized many fields including computer vision, natural language processing, and robotics.
Deep neural networks can learn complex patterns from large amounts of data.
The transformer architecture has become the foundation of modern language models.
Attention mechanisms allow models to focus on relevant parts of the input.
"""

    # Build the prompt
    prompt_parts = []

    # Add instruction
    prompt_parts.append("Please read the following information carefully and answer the question at the end.\n\n")

    # Add facts at different positions
    current_tokens = 50  # approximate tokens so far
    tokens_per_padding = 80  # approximate tokens per padding paragraph
    fact_interval = target_tokens // (len(facts) + 1)

    fact_idx = 0
    while current_tokens < target_tokens - 100:
        # Add padding
        prompt_parts.append(padding_paragraph)
        current_tokens += tokens_per_padding

        # Add a fact at intervals
        if fact_idx < len(facts) and current_tokens > fact_interval * (fact_idx + 1):
            prompt_parts.append(f"\n[Important Fact #{fact_idx + 1}]: {facts[fact_idx]}\n")
            current_tokens += 20
            fact_idx += 1

    # Add the question at the end
    prompt_parts.append("\n\nQuestion: Based on the information above, what is the capital of France and when was the Eiffel Tower built? Please answer briefly.\n\nAnswer:")

    return "".join(prompt_parts)


def test_chunked_prefill(num_gpu_blocks=10, input_len=8192, output_len=64):
    """Test chunked prefill with limited GPU blocks."""
    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")

    print(f"=" * 60)
    print(f"Chunked Prefill Test (Ping-Pong)")
    print(f"=" * 60)
    print(f"  target_input_len: ~{input_len} tokens")
    print(f"  num_gpu_blocks: {num_gpu_blocks}")
    print()

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=16 * 1024,
        max_num_batched_tokens=16 * 1024,
        enable_cpu_offload=True,
        num_gpu_blocks=num_gpu_blocks,
    )
    print()

    # Create meaningful prompt
    prompt = create_long_context_prompt(input_len)

    print(f"Running generation...")
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=output_len),  # low temperature for more deterministic output
        use_tqdm=False,
    )

    print()
    print(f"Output tokens: {len(outputs[0]['token_ids'])}")
    print(f"Output text:\n{outputs[0]['text']}")
    print()
    return outputs


def test_chunked_decode(num_gpu_blocks=10, input_len=8192, output_len=128):
    """Test chunked decode with limited GPU blocks."""
    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")

    print(f"=" * 60)
    print(f"Chunked Decode Test (Ping-Pong)")
    print(f"=" * 60)
    print(f"  target_input_len: ~{input_len} tokens")
    print(f"  output_len: {output_len} tokens")
    print(f"  num_gpu_blocks: {num_gpu_blocks}")
    print()

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=16 * 1024,
        max_num_batched_tokens=16 * 1024,
        enable_cpu_offload=True,
        num_gpu_blocks=num_gpu_blocks,
    )
    print()

    # Create meaningful prompt
    prompt = create_long_context_prompt(input_len)

    print(f"Running generation...")
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=output_len),
        use_tqdm=False,
    )

    print()
    print(f"Output tokens: {len(outputs[0]['token_ids'])}")
    print(f"Output text:\n{outputs[0]['text']}")
    print()
    return outputs


if __name__ == "__main__":
    # Parse arguments
    num_gpu_blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    input_len = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    output_len = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    test_chunked_prefill(num_gpu_blocks, input_len, output_len)