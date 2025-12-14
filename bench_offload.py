import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def bench_decode(llm, num_seqs, input_len, max_output_len):
    """Benchmark decode performance (original test)"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_output_tokens / t
    print(f"[Decode] Output: {total_output_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


def bench_prefill(llm, num_seqs, input_len):
    """Benchmark prefill performance"""
    seed(0)
    # Fixed length input, minimal output to focus on prefill
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_input_tokens = num_seqs * input_len
    throughput = total_input_tokens / t
    print(f"[Prefill] Input: {total_input_tokens}tok ({num_seqs}x{input_len}), Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


def main():
    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=128 * 1024,
        max_num_batched_tokens=128 * 1024,
        enable_cpu_offload=True,
        num_gpu_blocks=120,
        num_prefetch_blocks=4,
    )

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    print("=" * 60)
    print("Prefill Benchmark (CPU Offload)")
    print("=" * 60)
    # bench_prefill(llm, num_seqs=1, input_len=1024)
    # bench_prefill(llm, num_seqs=1, input_len=2048)
    # bench_prefill(llm, num_seqs=1, input_len=4096)
    bench_prefill(llm, num_seqs=1, input_len=16 * 1024)

    print("=" * 60)
    print("Decode Benchmark (CPU Offload)")
    print("=" * 60)
    bench_decode(llm, num_seqs=1, input_len=16 * 1024, max_output_len=128)
    # bench_decode(llm, num_seqs=1, input_len=2048, max_output_len=128)


if __name__ == "__main__":
    main()
