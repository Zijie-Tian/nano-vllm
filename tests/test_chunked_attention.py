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


def test_chunked_prefill(num_gpu_blocks=10, input_len=8192, output_len=16):
    """Test chunked prefill with limited GPU blocks."""
    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")

    total_blocks = (input_len + 255) // 256
    print(f"=" * 60)
    print(f"Chunked Prefill Test")
    print(f"=" * 60)
    print(f"  input_len: {input_len} tokens")
    print(f"  total_blocks: {total_blocks}")
    print(f"  num_gpu_blocks: {num_gpu_blocks}")
    print(f"  blocks_on_cpu: {max(0, total_blocks - num_gpu_blocks)}")
    print()

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=16 * 1024,  # 16K is enough for 8K test
        max_num_batched_tokens=16 * 1024,
        enable_cpu_offload=True,
        cpu_memory_gb=4.0,
        num_gpu_blocks=num_gpu_blocks,
    )

    print(f"LLM initialized:")
    print(f"  num_gpu_kvcache_blocks: {llm.model_runner.config.num_gpu_kvcache_blocks}")
    print(f"  num_cpu_kvcache_blocks: {llm.model_runner.config.num_cpu_kvcache_blocks}")
    print()

    # Create prompt with approximate token count
    prompt = "Hello " * (input_len // 2)

    print(f"Running generation...")
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=output_len),
        use_tqdm=True,
    )

    print()
    print(f"Output tokens: {len(outputs[0]['token_ids'])}")
    print(f"Output text (first 100 chars): {outputs[0]['text'][:100]}")
    print()
    return outputs


def test_chunked_decode(num_gpu_blocks=10, input_len=8192, output_len=64):
    """Test chunked decode with limited GPU blocks."""
    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")

    total_blocks = (input_len + 255) // 256
    print(f"=" * 60)
    print(f"Chunked Decode Test")
    print(f"=" * 60)
    print(f"  input_len: {input_len} tokens")
    print(f"  output_len: {output_len} tokens")
    print(f"  total_blocks: {total_blocks}")
    print(f"  num_gpu_blocks: {num_gpu_blocks}")
    print()

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=16 * 1024,
        max_num_batched_tokens=16 * 1024,
        enable_cpu_offload=True,
        cpu_memory_gb=4.0,
        num_gpu_blocks=num_gpu_blocks,
    )

    print(f"LLM initialized:")
    print(f"  num_gpu_kvcache_blocks: {llm.model_runner.config.num_gpu_kvcache_blocks}")
    print(f"  num_cpu_kvcache_blocks: {llm.model_runner.config.num_cpu_kvcache_blocks}")
    print()

    prompt = "Hello " * (input_len // 2)

    print(f"Running generation...")
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=output_len),
        use_tqdm=True,
    )

    print()
    print(f"Output tokens: {len(outputs[0]['token_ids'])}")
    print(f"Output text (first 100 chars): {outputs[0]['text'][:100]}")
    print()
    return outputs


if __name__ == "__main__":
    # Parse arguments
    num_gpu_blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    input_len = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    output_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    test_chunked_prefill(num_gpu_blocks, input_len, output_len)