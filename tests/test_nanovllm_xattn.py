"""
Test: NanoVLLM XAttention BSA Integration

Verify that nanovllm's XAttentionBSAPolicy works correctly with RULER-style tasks.
This is a simplified version of nanovllm's test_ruler.py for quick validation.

Usage:
    # Basic test (5 samples)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
        python tests/test_nanovllm_xattn.py

    # Compare FULL vs XATTN_BSA
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH \
        python tests/test_nanovllm_xattn.py --compare

    # Custom parameters
    python tests/test_nanovllm_xattn.py --threshold 0.95 --stride 8 --num-samples 10
"""

import os
import sys
import argparse
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure COMPASS and nanovllm are in path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "3rdparty/nanovllm"))

from nanovllm import LLM, SamplingParams
from nanovllm.config import SparsePolicyType


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = os.path.expanduser("~/models/Llama-3.1-8B-Instruct")
DEFAULT_MAX_MODEL_LEN = 65664  # 64K + buffer for output tokens
DEFAULT_MAX_NEW_TOKENS = 128

# Test data directory (use RULER benchmark data if available)
DATA_DIR = Path(__file__).parent.parent / "eval/RULER/scripts/benchmark_root"


# ============================================================
# Data Loading
# ============================================================

def load_samples(filepath: Path, num_samples: int = None) -> List[dict]:
    """Load samples from a JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            sample = json.loads(line)
            sample["_local_idx"] = i
            samples.append(sample)
    return samples


def find_ruler_data(task: str = "niah_single_1", seq_length: int = 32768) -> Path:
    """Find RULER benchmark data file."""
    # Search for existing benchmark data
    patterns = [
        DATA_DIR / f"xattn_stride16_llama3.1-8b-nanovllm/synthetic/{seq_length}/data/{task}/validation.jsonl",
        DATA_DIR / f"full_llama3.1-8b-nanovllm/synthetic/{seq_length}/data/{task}/validation.jsonl",
        DATA_DIR / f"*llama3.1*/{seq_length}/data/{task}/validation.jsonl",
    ]

    for pattern in patterns:
        if pattern.exists():
            return pattern
        # Try glob for wildcard patterns
        if "*" in str(pattern):
            matches = list(pattern.parent.parent.parent.parent.glob(f"*/synthetic/{seq_length}/data/{task}/validation.jsonl"))
            if matches:
                return matches[0]

    return None


# ============================================================
# Evaluation Functions (RULER Official Metrics)
# ============================================================

def string_match_all(output_text: str, expected_list: List[str]) -> float:
    """
    RULER official metric for NIAH tasks.
    Returns recall score (0.0 to 1.0): fraction of expected values found in output.
    """
    output_clean = output_text.replace('<|im_end|>', '').replace('\r', ' ').replace('\n', ' ')
    output_lower = output_clean.lower()

    if not expected_list:
        return 1.0

    found = sum(1.0 if exp.strip().lower() in output_lower else 0.0 for exp in expected_list)
    return found / len(expected_list)


# ============================================================
# Test Runner
# ============================================================

def create_llm(
    model_path: str,
    sparse_policy: SparsePolicyType,
    threshold: float = 0.9,
    stride: int = 16,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> LLM:
    """Create LLM instance with specified sparse policy."""
    llm_kwargs = {
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "kvcache_block_size": 1024,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "sparse_policy": sparse_policy,
        "enable_cpu_offload": True,
        "num_gpu_blocks": 2,
    }

    # XAttention-specific parameters (use nanovllm config field names)
    if sparse_policy == SparsePolicyType.XATTN_BSA:
        llm_kwargs["sparse_stride"] = stride
        llm_kwargs["sparse_threshold"] = threshold
        llm_kwargs["sparse_chunk_size"] = 16384
        llm_kwargs["sparse_use_triton"] = True

    return LLM(model_path, **llm_kwargs)


def run_test(
    llm: LLM,
    samples: List[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    verbose: bool = True,
) -> Dict:
    """Run test on samples and return results."""
    sampling_params = SamplingParams(
        temperature=0.01,  # Near-greedy
        max_tokens=max_new_tokens,
    )

    correct = 0
    total_score = 0.0
    results = []

    for sample in samples:
        idx = sample.get("index", sample["_local_idx"])
        prompt = sample["input"]
        expected = sample["outputs"]

        # Generate
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        output_text = outputs[0]["text"]

        # Evaluate
        score = string_match_all(output_text, expected)
        passed = score >= 0.5
        if passed:
            correct += 1
        total_score += score

        results.append({
            "index": idx,
            "expected": expected,
            "output": output_text[:100],
            "passed": passed,
            "score": score,
        })

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            exp_preview = str(expected[0])[:20] if expected else "N/A"
            out_preview = output_text[:30].replace('\n', ' ')
            print(f"  [{idx:3d}] {status} (score={score:.2f}) exp={exp_preview}... | out={out_preview}...")

    return {
        "correct": correct,
        "total": len(samples),
        "accuracy": correct / len(samples) if samples else 0.0,
        "avg_score": total_score / len(samples) if samples else 0.0,
        "results": results,
    }


def compare_policies(
    model_path: str,
    samples: List[dict],
    threshold: float,
    stride: int,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Compare FULL vs XATTN_BSA policies."""
    print("\n" + "=" * 60)
    print("Testing FULL policy (baseline)")
    print("=" * 60)

    llm_full = create_llm(model_path, SparsePolicyType.FULL)
    results_full = run_test(llm_full, samples, verbose=verbose)

    # Cleanup
    del llm_full
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print("\n" + "=" * 60)
    print(f"Testing XATTN_BSA policy (threshold={threshold}, stride={stride})")
    print("=" * 60)

    llm_xattn = create_llm(model_path, SparsePolicyType.XATTN_BSA, threshold, stride)
    results_xattn = run_test(llm_xattn, samples, verbose=verbose)

    # Cleanup
    del llm_xattn
    torch.cuda.empty_cache()
    gc.collect()

    return results_full, results_xattn


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test NanoVLLM XAttention BSA")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--task", type=str, default="niah_single_1",
                        help="RULER task name (default: niah_single_1)")
    parser.add_argument("--seq-length", type=int, default=32768,
                        help="Sequence length (default: 32768)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to test (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="XAttention threshold (default: 0.9)")
    parser.add_argument("--stride", type=int, default=16,
                        help="XAttention stride (default: 16)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare FULL vs XATTN_BSA")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Custom data file path")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode")

    args = parser.parse_args()

    # Find or validate data file
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = find_ruler_data(args.task, args.seq_length)

    if not data_file or not data_file.exists():
        print(f"ERROR: No data file found for task={args.task}, seq_length={args.seq_length}")
        print("Run RULER benchmark first to generate data:")
        print(f"  ./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic full --task {args.task}")
        sys.exit(1)

    print(f"Data file: {data_file}")

    # Load samples
    samples = load_samples(data_file, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    if args.compare:
        # Compare both policies
        results_full, results_xattn = compare_policies(
            args.model, samples, args.threshold, args.stride, verbose=not args.quiet
        )

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Policy':<15} {'Accuracy':<15} {'Avg Score':<15}")
        print("-" * 45)
        print(f"{'FULL':<15} {results_full['accuracy']*100:>6.1f}%        {results_full['avg_score']:.3f}")
        print(f"{'XATTN_BSA':<15} {results_xattn['accuracy']*100:>6.1f}%        {results_xattn['avg_score']:.3f}")
        print("=" * 60)

        # Check for degradation
        if results_xattn['accuracy'] < results_full['accuracy'] - 0.1:
            print("⚠️  WARNING: XATTN_BSA accuracy is significantly lower than FULL")
        elif results_xattn['accuracy'] >= results_full['accuracy']:
            print("✅ XATTN_BSA matches or exceeds FULL accuracy")

    else:
        # Test XATTN_BSA only
        print("\n" + "=" * 60)
        print(f"Testing XATTN_BSA (threshold={args.threshold}, stride={args.stride})")
        print("=" * 60)

        llm = create_llm(args.model, SparsePolicyType.XATTN_BSA, args.threshold, args.stride)
        results = run_test(llm, samples, verbose=not args.quiet)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Accuracy: {results['correct']}/{results['total']} ({results['accuracy']*100:.1f}%)")
        print(f"Avg Score: {results['avg_score']:.3f}")
        print("=" * 60)

        if results['accuracy'] >= 0.8:
            print("test_nanovllm_xattn: PASSED")
        else:
            print(f"test_nanovllm_xattn: FAILED (accuracy={results['accuracy']*100:.1f}%)")
            sys.exit(1)


if __name__ == "__main__":
    main()
