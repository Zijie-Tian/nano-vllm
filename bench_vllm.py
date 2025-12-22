import os
os.environ["VLLM_USE_V1"] = "1"
import time
from random import randint, seed
from vllm import LLM, SamplingParams


def bench_decode(llm, num_seqs, input_len, output_len):
    """Benchmark decode performance (original test)"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
    prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_output_tokens = num_seqs * output_len
    throughput = total_output_tokens / t
    print(f"[Decode] Input: {num_seqs}x{input_len}tok, Output: {total_output_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


def bench_prefill(llm, num_seqs, input_len):
    """Benchmark prefill performance"""
    seed(0)
    # Fixed length input, minimal output to focus on prefill
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_input_tokens = num_seqs * input_len
    throughput = total_input_tokens / t
    print(f"[Prefill] Input: {total_input_tokens}tok ({num_seqs}x{input_len}), Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


def main():
    path = os.path.expanduser("~/models/Qwen3-0.6B/")
    # Note: Qwen3-0.6B max_position_embeddings = 40960, cannot exceed this
    max_len = 40960
    llm = LLM(path, enforce_eager=False, max_model_len=max_len, max_num_seqs=128, gpu_memory_utilization=0.9)

    # Warmup
    llm.generate([dict(prompt_token_ids=[0])], SamplingParams())

    print("=" * 60)
    print("Prefill Benchmark")
    print("=" * 60)
    # bench_prefill(llm, num_seqs=1, input_len=1024)
    # bench_prefill(llm, num_seqs=1, input_len=2048)
    bench_prefill(llm, num_seqs=1, input_len=max_len - 1)
    # bench_prefill(llm, num_seqs=16, input_len=1024)
    # bench_prefill(llm, num_seqs=64, input_len=1024)

    print("=" * 60)
    print("Decode Benchmark")
    print("=" * 60)
    # bench_decode(llm, num_seqs=1, input_len=1024, output_len=1024)
    bench_decode(llm, num_seqs=1, input_len=max_len - 128, output_len=128)  # input + output <= max_len


if __name__ == "__main__":
    main()
