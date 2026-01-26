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
    decode_throughput = decode_tokens / t

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

    parser = argparse.ArgumentParser(description="Benchmark nanovllm GPU performance")
    parser.add_argument("--model", type=str, default="~/models/Llama-3.1-8B-Instruct",
                        help="Model path (default: ~/models/Llama-3.1-8B-Instruct)")
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=64, help="Output length for decode benchmark (default: 64)")
    parser.add_argument("--max-len", type=int, default=32*1024, help="Max model length (default: 32K)")
    parser.add_argument("--bench-decode", action="store_true", help="Run decode benchmark (default: prefill only)")
    parser.add_argument("--bench-all", action="store_true", help="Run both prefill and decode benchmarks")
    # Sparse policy option (GPU-only mode now supports policy routing)
    parser.add_argument("--policy", type=str, default=None,
                        choices=["full", "xattn"],
                        help="Sparse policy: full (FullAttention), xattn (XAttention+BSA)")
    parser.add_argument("--enable-policy", action="store_true",
                        help="Enable sparse policy routing (FullAttentionPolicy by default)")
    parser.add_argument("--gpu-util", type=float, default=0.9,
                        help="GPU memory utilization (default: 0.9)")
    args = parser.parse_args()

    path = os.path.expanduser(args.model)
    max_len = args.max_len

    # Configure sparse policy
    if args.policy == "xattn":
        sparse_policy = SparsePolicyType.XATTN_BSA
        print(f"\n[nanovllm GPU + XAttention BSA] max_len={max_len}")
    elif args.policy == "full" or args.enable_policy:
        sparse_policy = SparsePolicyType.FULL
        print(f"\n[nanovllm GPU + Policy] sparse_policy=FULL, max_len={max_len}")
    else:
        sparse_policy = None
        print(f"\n[nanovllm GPU] max_len={max_len}")

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        sparse_policy=sparse_policy,
        gpu_memory_utilization=args.gpu_util,
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
        print("Prefill Benchmark (nanovllm GPU)")
        print("=" * 60)
        bench_prefill(llm, num_seqs=1, input_len=prefill_input_len)

    if run_decode:
        print("\n" + "=" * 60)
        print("Decode Benchmark (nanovllm GPU)")
        print("=" * 60)
        bench_decode(llm, num_seqs=1, input_len=decode_input_len, output_len=args.output_len)


if __name__ == "__main__":
    main()
