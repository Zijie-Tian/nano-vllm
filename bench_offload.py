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
    parser.add_argument("--no-sparse", action="store_true", help="Disable sparse attention (baseline)")
    parser.add_argument("--topk", type=int, default=8, help="Top-K blocks for Quest")
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens (default: max_len - 1 for prefill, max_len - output_len for decode)")
    parser.add_argument("--output-len", type=int, default=128, help="Output length in tokens")
    args = parser.parse_args()

    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    # Note: Qwen3-4B-Instruct-2507 max_position_embeddings = 262144
    max_len = 32 * 1024  # 128K tokens

    # Setup policy configuration
    if not args.no_sparse:
        prefill_policy = "full"   # Full attention for prefill
        decode_policy = "quest"   # Quest Top-K for decode
        print(f"\n[Quest Sparse Attention] prefill={prefill_policy}, decode={decode_policy}, topk={args.topk}")
    else:
        prefill_policy = "full"   # Full attention for both phases
        decode_policy = "full"
        print("\n[Full Attention] No sparse policy (baseline)")

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        enable_cpu_offload=True,
        num_gpu_blocks=6,  # Small GPU buffer for offload testing
        prefill_policy=prefill_policy,
        decode_policy=decode_policy,
        sparse_topk_blocks=args.topk,
        sparse_threshold_blocks=4,
    )

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # Default input lengths based on max_len
    prefill_input_len = args.input_len if args.input_len else max_len - 1
    decode_input_len = args.input_len if args.input_len else max_len - args.output_len

    print("=" * 60)
    print("Prefill Benchmark (CPU Offload)")
    print("=" * 60)
    bench_prefill(llm, num_seqs=1, input_len=prefill_input_len)

    # print("=" * 60)
    # print("Decode Benchmark (CPU Offload)")
    # print("=" * 60)
    # bench_decode(llm, num_seqs=1, input_len=decode_input_len, output_len=args.output_len)


if __name__ == "__main__":
    main()
