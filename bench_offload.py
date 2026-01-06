import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def bench_decode(llm, num_seqs, input_len, output_len):
    """Benchmark decode performance"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    # Calculate metrics
    prefill_tokens = num_seqs * input_len
    decode_tokens = num_seqs * output_len

    # Approximate: assume prefill takes ~input_len/prefill_speed, rest is decode
    # For more accurate measurement, we'd need internal timing
    decode_throughput = decode_tokens / t  # This includes prefill time, so it's a lower bound

    print(f"[Decode] Input: {num_seqs}x{input_len}tok, Output: {decode_tokens}tok, Time: {t:.2f}s")
    print(f"         Throughput: {decode_throughput:.2f} tok/s (includes prefill overhead)")


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
    from nanovllm.config import SparsePolicyType

    parser = argparse.ArgumentParser(description="Benchmark CPU offload performance")
    parser.add_argument("--enable-quest", action="store_true", help="Enable Quest sparse attention for decode")
    parser.add_argument("--topk", type=int, default=16, help="Top-K blocks for Quest (default: 16)")
    parser.add_argument("--threshold", type=int, default=4, help="Apply sparse only when blocks > threshold (default: 4)")
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=64, help="Output length for decode benchmark (default: 64)")
    parser.add_argument("--num-gpu-blocks", type=int, default=6, help="Number of GPU blocks (default: 6)")
    parser.add_argument("--max-len", type=int, default=32*1024, help="Max model length (default: 32K)")
    parser.add_argument("--bench-decode", action="store_true", help="Run decode benchmark (default: prefill only)")
    parser.add_argument("--bench-all", action="store_true", help="Run both prefill and decode benchmarks")
    args = parser.parse_args()

    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    max_len = args.max_len

    # Setup policy configuration
    if args.enable_quest:
        sparse_policy = SparsePolicyType.QUEST
        print(f"\n[Quest Sparse Attention] topk={args.topk}, threshold={args.threshold}")
    else:
        sparse_policy = SparsePolicyType.FULL
        print("\n[Full Attention] baseline (no sparse)")

    print(f"[Config] max_len={max_len}, num_gpu_blocks={args.num_gpu_blocks}")

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        enable_cpu_offload=True,
        num_gpu_blocks=args.num_gpu_blocks,
        sparse_policy=sparse_policy,
        sparse_topk_blocks=args.topk,
        sparse_threshold_blocks=args.threshold,
    )

    # Warmup
    print("\nWarming up...")
    llm.generate(["Benchmark warmup: "], SamplingParams(max_tokens=10))

    # Default input lengths
    prefill_input_len = args.input_len if args.input_len else max_len - 1
    decode_input_len = args.input_len if args.input_len else max_len - args.output_len

    # Determine which benchmarks to run
    run_prefill = not args.bench_decode or args.bench_all
    run_decode = args.bench_decode or args.bench_all

    if run_prefill:
        print("\n" + "=" * 60)
        print("Prefill Benchmark (CPU Offload)")
        print("=" * 60)
        bench_prefill(llm, num_seqs=1, input_len=prefill_input_len)

    if run_decode:
        print("\n" + "=" * 60)
        print("Decode Benchmark (CPU Offload)")
        print("=" * 60)
        bench_decode(llm, num_seqs=1, input_len=decode_input_len, output_len=args.output_len)


if __name__ == "__main__":
    main()
