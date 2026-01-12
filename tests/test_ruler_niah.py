"""
RULER NIAH benchmark test for LLM.

Tests: Long context retrieval capability using pre-generated RULER benchmark data.
The NIAH (Needle-In-A-Haystack) task tests the model's ability to retrieve a
specific magic number from a large context (~32K tokens).

Usage:
    # Test all samples with CPU offload
    python tests/test_ruler_niah.py --enable-offload

    # Test specific samples
    python tests/test_ruler_niah.py --sample-indices 0,1,2 --enable-offload

    # Test with custom model
    python tests/test_ruler_niah.py --model /path/to/model --enable-offload

    # Group mode: test in batches with separate LLM initialization per group
    python tests/test_ruler_niah.py --enable-offload --group-size 5
"""

import os
os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

from nanovllm import LLM, SamplingParams
from utils import check_needle_answer


# ============================================================
# Constants
# ============================================================

DEFAULT_DATA_FILE = Path(__file__).parent / "data/ruler_niah/niah_single_1_32k.jsonl"
DEFAULT_MODEL = os.path.expanduser("~/models/Llama-3.1-8B-Instruct")
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_NEW_TOKENS = 50


# ============================================================
# Data Loading
# ============================================================

def load_ruler_samples(filepath: Path, indices: Optional[List[int]] = None) -> List[dict]:
    """
    Load RULER NIAH samples from a JSONL file.

    Args:
        filepath: Path to the JSONL file
        indices: Optional list of sample indices to load. If None, load all.

    Returns:
        List of sample dicts with keys: index, input, outputs, length
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please copy RULER NIAH data to this location. See docs/ruler_niah_standalone_test.md"
        )

    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if indices is None or i in indices:
                sample = json.loads(line)
                samples.append(sample)

    if not samples:
        raise ValueError(f"No samples loaded from {filepath}")

    return samples


def count_samples(filepath: Path) -> int:
    """Count total samples in JSONL file."""
    with open(filepath) as f:
        return sum(1 for _ in f)


# ============================================================
# Test Function
# ============================================================

def run_ruler_niah_test(
    model_path: str,
    data_file: Path,
    sample_indices: Optional[List[int]] = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_cpu_offload: bool = False,
    num_gpu_blocks: int = 4,
    block_size: int = 1024,
    gpu_utilization: float = 0.9,
    enforce_eager: bool = True,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Run RULER NIAH test on loaded samples.

    Args:
        model_path: Path to the model
        data_file: Path to JSONL data file
        sample_indices: List of sample indices to test (None = all)
        max_model_len: Maximum model context length
        max_new_tokens: Maximum tokens to generate
        enable_cpu_offload: Enable CPU offload mode
        num_gpu_blocks: Number of GPU blocks for offload
        block_size: KV cache block size
        gpu_utilization: GPU memory utilization fraction
        enforce_eager: Disable CUDA graphs
        verbose: Print detailed output

    Returns:
        (correct, total): Number of correct and total samples
    """
    # Load samples
    samples = load_ruler_samples(data_file, sample_indices)
    total = len(samples)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RULER NIAH Test")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Data file: {data_file}")
        print(f"Samples: {total}")
        print(f"Max model len: {max_model_len}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"CPU offload: {enable_cpu_offload}")
        if enable_cpu_offload:
            print(f"  num_gpu_blocks: {num_gpu_blocks}")
            print(f"  block_size: {block_size}")
        print(f"Enforce eager: {enforce_eager}")
        print(f"{'='*60}\n")

    # Check max_model_len vs data length
    max_data_len = max(s.get("length", 0) for s in samples)
    if max_model_len < max_data_len:
        print(f"WARNING: max_model_len ({max_model_len}) < max data length ({max_data_len})")
        print(f"         This may cause truncation or errors.\n")

    # Initialize LLM
    if verbose:
        print("Initializing LLM...")

    llm_kwargs = {
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_utilization,
        "kvcache_block_size": block_size,
        "enable_cpu_offload": enable_cpu_offload,
    }

    if enable_cpu_offload:
        llm_kwargs["num_gpu_blocks"] = num_gpu_blocks

    llm = LLM(model_path, **llm_kwargs)

    # Sampling params
    # Note: nano-vllm doesn't support greedy (temperature=0), use low temperature instead
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for near-deterministic output
        max_tokens=max_new_tokens,
    )

    # Test each sample
    correct = 0
    results = []

    for i, sample in enumerate(samples):
        sample_idx = sample.get("index", i)
        prompt = sample["input"]
        expected = sample["outputs"][0]
        data_len = sample.get("length", "unknown")

        if verbose:
            print(f"\nSample {sample_idx}: Expected={expected}, Length={data_len}")

        # Generate
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        output_text = outputs[0]["text"]
        output_tokens = outputs[0]["token_ids"]

        # Check result
        passed = check_needle_answer(output_text, expected)
        if passed:
            correct += 1

        results.append({
            "index": sample_idx,
            "expected": expected,
            "output": output_text,
            "passed": passed,
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            output_preview = output_text[:100].replace('\n', ' ')
            print(f"  Output ({len(output_tokens)} tokens): {output_preview}...")
            print(f"  Status: {status}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results: {correct}/{total} PASSED ({100*correct/total:.1f}%)")
        print(f"{'='*60}\n")

        if correct < total:
            print("Failed samples:")
            for r in results:
                if not r["passed"]:
                    print(f"  Sample {r['index']}: expected={r['expected']}, got={r['output'][:50]}...")

    return correct, total


# ============================================================
# Grouped Test Function
# ============================================================

def run_grouped_test(
    model_path: str,
    data_file: Path,
    group_size: int = 5,
    total_samples: Optional[int] = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_cpu_offload: bool = False,
    num_gpu_blocks: int = 4,
    block_size: int = 1024,
    gpu_utilization: float = 0.9,
    enforce_eager: bool = True,
) -> Tuple[int, int, List[dict]]:
    """
    Run RULER NIAH test in groups, with separate LLM initialization per group.

    This mode is useful for:
    - Avoiding state accumulation issues
    - Testing LLM initialization stability
    - Running large-scale tests with memory cleanup between groups

    Args:
        model_path: Path to the model
        data_file: Path to JSONL data file
        group_size: Number of samples per group
        total_samples: Total samples to test (None = all in file)
        Other args: Same as run_ruler_niah_test

    Returns:
        (total_correct, total_tested, group_results): Results summary
    """
    import time
    import gc
    import torch

    # Count total samples in file
    file_sample_count = count_samples(data_file)
    if total_samples is None:
        total_samples = file_sample_count
    else:
        total_samples = min(total_samples, file_sample_count)

    num_groups = (total_samples + group_size - 1) // group_size

    print(f"\n{'='*60}")
    print(f"RULER NIAH Grouped Test")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data file: {data_file}")
    print(f"Total samples: {total_samples}")
    print(f"Group size: {group_size}")
    print(f"Number of groups: {num_groups}")
    print(f"CPU offload: {enable_cpu_offload}")
    print(f"{'='*60}\n")

    total_correct = 0
    total_tested = 0
    group_results = []
    all_failed = []

    test_start_time = time.time()

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, total_samples)
        sample_indices = list(range(start_idx, end_idx))

        print(f"\n{'='*60}")
        print(f"Group {group_idx + 1}/{num_groups}: Samples {start_idx}-{end_idx - 1}")
        print(f"{'='*60}")

        group_start_time = time.time()

        # Run test for this group
        correct, tested = run_ruler_niah_test(
            model_path=model_path,
            data_file=data_file,
            sample_indices=sample_indices,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            enable_cpu_offload=enable_cpu_offload,
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            gpu_utilization=gpu_utilization,
            enforce_eager=enforce_eager,
            verbose=True,
        )

        group_time = time.time() - group_start_time

        total_correct += correct
        total_tested += tested

        group_result = {
            "group": group_idx + 1,
            "samples": f"{start_idx}-{end_idx - 1}",
            "correct": correct,
            "total": tested,
            "accuracy": 100 * correct / tested if tested > 0 else 0,
            "time": group_time,
        }
        group_results.append(group_result)

        print(f"\nGroup {group_idx + 1} Summary: {correct}/{tested} PASSED ({group_result['accuracy']:.1f}%) in {group_time:.1f}s")

        # Force cleanup between groups
        gc.collect()
        torch.cuda.empty_cache()

        # Small delay to ensure port is released
        if group_idx < num_groups - 1:
            time.sleep(3)

    total_time = time.time() - test_start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nGroup Results:")
    print(f"{'Group':<8} {'Samples':<12} {'Result':<12} {'Accuracy':<10} {'Time':<10}")
    print(f"{'-'*52}")
    for r in group_results:
        print(f"{r['group']:<8} {r['samples']:<12} {r['correct']}/{r['total']:<9} {r['accuracy']:.1f}%{'':<5} {r['time']:.1f}s")

    print(f"{'-'*52}")
    overall_accuracy = 100 * total_correct / total_tested if total_tested > 0 else 0
    print(f"{'TOTAL':<8} {'0-' + str(total_tested-1):<12} {total_correct}/{total_tested:<9} {overall_accuracy:.1f}%{'':<5} {total_time:.1f}s")
    print(f"{'='*60}\n")

    return total_correct, total_tested, group_results


# ============================================================
# CLI Entry Point
# ============================================================

def parse_indices(s: str) -> List[int]:
    """Parse comma-separated indices like '0,1,2' or range like '0-4'."""
    if not s:
        return None
    indices = []
    for part in s.split(','):
        if '-' in part:
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RULER NIAH benchmark test for long context LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all samples with CPU offload (recommended for 24GB GPUs)
  python tests/test_ruler_niah.py --enable-offload

  # Test specific samples
  python tests/test_ruler_niah.py --sample-indices 0,1,2 --enable-offload

  # Test with CUDA graph enabled
  python tests/test_ruler_niah.py --enable-offload --use-cuda-graph
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help=f"Path to JSONL data file (default: {DEFAULT_DATA_FILE})"
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Sample indices to test (e.g., '0,1,2' or '0-4'). Default: all"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum model context length (default: {DEFAULT_MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--enable-offload",
        action="store_true",
        help="Enable CPU offload mode (required for 24GB GPUs with 32K context)"
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=4,
        help="Number of GPU blocks for CPU offload (default: 4)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="KV cache block size (default: 1024)"
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Force eager execution, disable CUDA graphs (default: True)"
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Enable CUDA graph (overrides --enforce-eager)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode, only print final result"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Enable grouped testing mode with specified group size. Each group initializes LLM separately. (default: 0 = disabled)"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=0,
        help="Total number of samples to test in group mode (default: 0 = all samples in file)"
    )

    args = parser.parse_args()

    # Process arguments
    sample_indices = parse_indices(args.sample_indices)
    enforce_eager = not args.use_cuda_graph
    verbose = not args.quiet

    # Check if group mode is enabled
    if args.group_size > 0:
        # Grouped testing mode
        total_samples = args.total_samples if args.total_samples > 0 else None
        correct, total, _ = run_grouped_test(
            model_path=os.path.expanduser(args.model),
            data_file=Path(args.data_file),
            group_size=args.group_size,
            total_samples=total_samples,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            enable_cpu_offload=args.enable_offload,
            num_gpu_blocks=args.num_gpu_blocks,
            block_size=args.block_size,
            gpu_utilization=args.gpu_utilization,
            enforce_eager=enforce_eager,
        )
    else:
        # Standard testing mode
        correct, total = run_ruler_niah_test(
            model_path=os.path.expanduser(args.model),
            data_file=Path(args.data_file),
            sample_indices=sample_indices,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            enable_cpu_offload=args.enable_offload,
            num_gpu_blocks=args.num_gpu_blocks,
            block_size=args.block_size,
            gpu_utilization=args.gpu_utilization,
            enforce_eager=enforce_eager,
            verbose=verbose,
        )

    # Final status
    if correct == total:
        print("test_ruler_niah: PASSED")
    else:
        print(f"test_ruler_niah: FAILED ({correct}/{total})")
        exit(1)
