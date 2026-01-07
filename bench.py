import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.config import SparsePolicyType


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


def bench_prefill(llm, num_seqs, input_len, label=""):
    """Benchmark prefill performance. Returns throughput."""
    seed(0)
    # Fixed length input, minimal output to focus on prefill
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_input_tokens = num_seqs * input_len
    throughput = total_input_tokens / t
    label_str = f" ({label})" if label else ""
    print(f"[Prefill{label_str}] Input: {total_input_tokens}tok ({num_seqs}x{input_len}), Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    return throughput


def create_llm(path, max_len, enable_minference=False, minference_budget=0.3,
               minference_vertical=1000, minference_slash=6096,
               gpu_utilization=0.8):
    """Create LLM with specified configuration."""
    kwargs = {
        "enforce_eager": True,  # MInference uses Triton, not compatible with CUDA graphs
        "max_model_len": max_len,
        "max_num_batched_tokens": max_len,
        "gpu_memory_utilization": gpu_utilization,
    }
    if enable_minference:
        kwargs["sparse_policy"] = SparsePolicyType.MINFERENCE
        kwargs["minference_adaptive_budget"] = minference_budget
        kwargs["minference_vertical_size"] = minference_vertical
        kwargs["minference_slash_size"] = minference_slash

    return LLM(path, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark nanovllm GPU performance")
    parser.add_argument("--input-len", type=int, default=None, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=64, help="Output length for decode benchmark (default: 64)")
    parser.add_argument("--max-len", type=int, default=32*1024, help="Max model length (default: 32K)")
    parser.add_argument("--bench-decode", action="store_true", help="Run decode benchmark (default: prefill only)")
    parser.add_argument("--bench-all", action="store_true", help="Run both prefill and decode benchmarks")
    parser.add_argument("--enable-minference", action="store_true", help="Enable MInference sparse prefill")
    parser.add_argument("--minference-budget", type=float, default=0.3, help="MInference adaptive budget (default: 0.3, use 0 for fixed mode)")
    parser.add_argument("--minference-vertical", type=int, default=1000, help="Fixed vertical_size (only used when budget=0)")
    parser.add_argument("--minference-slash", type=int, default=6096, help="Fixed slash_size (only used when budget=0)")
    parser.add_argument("--gpu-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs MInference (runs both)")
    args = parser.parse_args()

    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    max_len = args.max_len

    # Default input lengths
    prefill_input_len = args.input_len if args.input_len else max_len - 1
    decode_input_len = args.input_len if args.input_len else max_len - args.output_len

    # Determine which benchmarks to run
    run_prefill = not args.bench_decode or args.bench_all
    run_decode = args.bench_decode or args.bench_all

    # Convert budget=0 to None for fixed mode
    minference_budget = args.minference_budget if args.minference_budget > 0 else None

    if args.compare:
        # Compare baseline vs MInference using subprocesses to avoid NCCL issues
        import subprocess
        import sys

        print(f"\n{'='*60}")
        print(f"Baseline vs MInference Comparison")
        print(f"Input length: {prefill_input_len} tokens")
        if minference_budget is not None:
            print(f"MInference mode: adaptive (budget={minference_budget}, {minference_budget*100:.0f}% compute)")
        else:
            print(f"MInference mode: fixed (vertical={args.minference_vertical}, slash={args.minference_slash})")
        print(f"{'='*60}")

        # Get PYTHONPATH for subprocess
        pythonpath = os.environ.get("PYTHONPATH", "")

        # Run baseline in subprocess
        print(f"\n[1/2] Running baseline (FULL attention)...")
        cmd_baseline = [
            sys.executable, __file__,
            "--input-len", str(prefill_input_len),
            "--max-len", str(max_len),
            "--gpu-utilization", str(args.gpu_utilization),
        ]
        env = os.environ.copy()
        result = subprocess.run(cmd_baseline, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return

        # Parse baseline throughput
        baseline_throughput = None
        for line in result.stdout.split('\n'):
            if "Throughput:" in line and "tok/s" in line:
                # Extract throughput value
                import re
                match = re.search(r'Throughput:\s*([\d.]+)tok/s', line)
                if match:
                    baseline_throughput = float(match.group(1))

        # Run MInference in subprocess
        if minference_budget is not None:
            print(f"\n[2/2] Running MInference (budget={minference_budget})...")
        else:
            print(f"\n[2/2] Running MInference (vertical={args.minference_vertical}, slash={args.minference_slash})...")
        cmd_minference = [
            sys.executable, __file__,
            "--input-len", str(prefill_input_len),
            "--max-len", str(max_len),
            "--gpu-utilization", str(args.gpu_utilization),
            "--enable-minference",
            "--minference-budget", str(args.minference_budget),
            "--minference-vertical", str(args.minference_vertical),
            "--minference-slash", str(args.minference_slash),
        ]
        result = subprocess.run(cmd_minference, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return

        # Parse MInference throughput
        minference_throughput = None
        for line in result.stdout.split('\n'):
            if "Throughput:" in line and "tok/s" in line:
                import re
                match = re.search(r'Throughput:\s*([\d.]+)tok/s', line)
                if match:
                    minference_throughput = float(match.group(1))

        # Comparison
        if baseline_throughput and minference_throughput:
            print(f"\n{'='*60}")
            print(f"Results Summary")
            print(f"{'='*60}")
            print(f"Baseline:   {baseline_throughput:,.0f} tok/s")
            print(f"MInference: {minference_throughput:,.0f} tok/s")
            speedup = minference_throughput / baseline_throughput
            if speedup >= 1.0:
                print(f"Speedup:    {speedup:.2f}x faster")
            else:
                print(f"Slowdown:   {1/speedup:.2f}x slower")
            print(f"{'='*60}")
        else:
            print("Failed to parse throughput values")

    else:
        # Single run mode
        mode = "MInference" if args.enable_minference else "GPU"
        print(f"\n[nanovllm {mode}] max_len={max_len}")
        if args.enable_minference:
            if minference_budget is not None:
                print(f"MInference mode: adaptive (budget={minference_budget})")
            else:
                print(f"MInference mode: fixed (vertical={args.minference_vertical}, slash={args.minference_slash})")

        llm = create_llm(path, max_len, enable_minference=args.enable_minference,
                        minference_budget=minference_budget,
                        minference_vertical=args.minference_vertical,
                        minference_slash=args.minference_slash,
                        gpu_utilization=args.gpu_utilization)

        # Warmup
        print("\nWarming up...")
        llm.generate(["Benchmark warmup: "], SamplingParams(max_tokens=10))

        if run_prefill:
            print("\n" + "=" * 60)
            print(f"Prefill Benchmark (nanovllm {mode})")
            print("=" * 60)
            bench_prefill(llm, num_seqs=1, input_len=prefill_input_len)

        if run_decode:
            print("\n" + "=" * 60)
            print(f"Decode Benchmark (nanovllm {mode})")
            print("=" * 60)
            bench_decode(llm, num_seqs=1, input_len=decode_input_len, output_len=args.output_len)


if __name__ == "__main__":
    main()
