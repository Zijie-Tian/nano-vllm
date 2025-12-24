import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def bench_decode(llm, num_seqs, input_len, output_len):
    """Benchmark decode performance (original test)"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)

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

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_input_tokens = num_seqs * input_len
    throughput = total_input_tokens / t
    print(f"[Prefill] Input: {total_input_tokens}tok ({num_seqs}x{input_len}), Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=128, help="Output length in tokens")
    args = parser.parse_args()

    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    # Note: Qwen3-4B-Instruct-2507 max_position_embeddings = 262144
    max_len = 131072  # 128K tokens
    llm = LLM(path, enforce_eager=False, max_model_len=max_len, max_num_batched_tokens=max_len)

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # Default input lengths based on max_len
    prefill_input_len = args.input_len if args.input_len else max_len - 1
    decode_input_len = args.input_len if args.input_len else max_len - args.output_len

    print("=" * 60)
    print("Prefill Benchmark (GPU)")
    print("=" * 60)
    bench_prefill(llm, num_seqs=1, input_len=prefill_input_len)

    # print("=" * 60)
    # print("Decode Benchmark (GPU)")
    # print("=" * 60)
    # bench_decode(llm, num_seqs=1, input_len=decode_input_len, output_len=args.output_len)


if __name__ == "__main__":
    main()
