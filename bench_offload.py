import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.observer import InferenceObserver


def bench_decode(llm, num_seqs, input_len, output_len):
    """Benchmark decode performance"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    # Get metrics from InferenceObserver
    ttft_ms = InferenceObserver.ttft / 1e6
    tpot_ms = InferenceObserver.tpot / 1e6

    # Calculate throughput from observer metrics
    decode_tokens = num_seqs * output_len
    decode_throughput = 1000.0 / tpot_ms if tpot_ms > 0 else 0  # tokens/s per sequence

    print(f"[Decode] Input: {num_seqs}x{input_len}tok, Output: {decode_tokens}tok, Time: {t:.2f}s")
    print(f"         TTFT: {ttft_ms:.2f}ms, TPOT: {tpot_ms:.2f}ms")
    print(f"         Decode Throughput: {decode_throughput:.2f} tok/s (from observer)")


def bench_prefill(llm, num_seqs, input_len):
    """Benchmark prefill performance"""
    seed(0)
    # Fixed length input, minimal output to focus on prefill
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    # Get TTFT from InferenceObserver
    ttft_ms = InferenceObserver.ttft / 1e6
    ttft_s = ttft_ms / 1000.0

    total_input_tokens = num_seqs * input_len
    # Use observer TTFT for accurate prefill throughput
    throughput_observer = total_input_tokens / ttft_s if ttft_s > 0 else 0
    throughput_external = total_input_tokens / t

    print(f"[Prefill] Input: {total_input_tokens}tok ({num_seqs}x{input_len})")
    print(f"          External Time: {t:.2f}s, Throughput: {throughput_external:.2f}tok/s")
    print(f"          Observer TTFT: {ttft_ms:.2f}ms, Throughput: {throughput_observer:.2f}tok/s")


def main():
    import argparse
    from nanovllm.config import SparsePolicyType

    parser = argparse.ArgumentParser(description="Benchmark CPU offload performance")
    parser.add_argument("--model", type=str, default="~/models/Llama-3.1-8B-Instruct",
                        help="Model path (default: ~/models/Llama-3.1-8B-Instruct)")
    # Sparse policy selection (mutually exclusive)
    sparse_group = parser.add_mutually_exclusive_group()
    sparse_group.add_argument("--enable-quest", action="store_true",
                              help="Enable Quest sparse attention (decode only, prefill uses full)")
    sparse_group.add_argument("--enable-xattn", action="store_true",
                              help="Enable XAttention BSA (prefill only, decode uses full)")
    # Quest parameters
    parser.add_argument("--topk", type=int, default=16, help="Top-K blocks for Quest (default: 16)")
    parser.add_argument("--threshold", type=int, default=4, help="Apply sparse only when blocks > threshold (default: 4)")
    # XAttention parameters
    parser.add_argument("--xattn-threshold", type=float, default=0.95,
                        help="XAttention cumulative attention threshold (default: 0.95)")
    parser.add_argument("--xattn-stride", type=int, default=8,
                        help="XAttention Q/K downsampling stride (default: 8)")
    # General parameters
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=64, help="Output length for decode benchmark (default: 64)")
    parser.add_argument("--num-gpu-blocks", type=int, default=4, help="Number of GPU blocks (default: 4)")
    parser.add_argument("--block-size", type=int, default=1024, help="KV cache block size (default: 1024)")
    parser.add_argument("--max-len", type=int, default=32*1024, help="Max model length (default: 32K)")
    parser.add_argument("--bench-decode", action="store_true", help="Run decode benchmark (default: prefill only)")
    parser.add_argument("--bench-all", action="store_true", help="Run both prefill and decode benchmarks")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA Graphs (use eager mode)")
    args = parser.parse_args()

    path = os.path.expanduser(args.model)
    max_len = args.max_len

    # Setup policy configuration
    if args.enable_quest:
        sparse_policy = SparsePolicyType.QUEST
        print(f"\n[Quest Sparse Attention] decode: Quest (topk={args.topk}, threshold={args.threshold}), prefill: Full")
    elif args.enable_xattn:
        sparse_policy = SparsePolicyType.XATTN_BSA
        print(f"\n[XAttention BSA] prefill: XAttn (tau={args.xattn_threshold}, stride={args.xattn_stride}), decode: Full")
    else:
        sparse_policy = SparsePolicyType.FULL
        print("\n[Full Attention] baseline (no sparse)")

    print(f"[Config] max_len={max_len}, num_gpu_blocks={args.num_gpu_blocks}, block_size={args.block_size}")

    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        enable_cpu_offload=True,
        num_gpu_blocks=args.num_gpu_blocks,
        kvcache_block_size=args.block_size,
        sparse_policy=sparse_policy,
        # Quest parameters
        sparse_topk_blocks=args.topk,
        sparse_threshold_blocks=args.threshold,
        # XAttention parameters
        sparse_threshold=args.xattn_threshold,
        sparse_stride=args.xattn_stride,
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
