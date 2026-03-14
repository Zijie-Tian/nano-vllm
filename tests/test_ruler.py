"""
RULER benchmark comprehensive test for LLM.

Tests multiple RULER tasks:
- NIAH (Needle-In-A-Haystack): single, multikey, multiquery, multivalue
- QA (Question Answering): qa_1, qa_2
- CWE (Common Word Extraction)
- FWE (Frequent Word Extraction)
- VT (Variable Tracking)

Usage:
    # Test all datasets with 2 samples each (debug mode)
    python tests/test_ruler.py --enable-offload --num-samples 2

    # Test specific datasets
    python tests/test_ruler.py --enable-offload --datasets niah_single_1,qa_1

    # Test all samples in all datasets
    python tests/test_ruler.py --enable-offload

    # Test specific sample indices (comma-separated)
    python tests/test_ruler.py --enable-offload --datasets niah_single_1 --sample-indices 28,33,40

    # Single-sample mode: reinitialize LLM for each sample (avoids state leakage)
    python tests/test_ruler.py --enable-offload --datasets niah_single_1 --fresh-llm

    # JSON output mode for scripting
    python tests/test_ruler.py --enable-offload --datasets niah_single_1 --json-output
"""

import os

os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"

import argparse
import json
import gc
import time
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from nanovllm import LLM, SamplingParams
from nanovllm.utils.density_observer import DensityObserver


# ============================================================
# Constants
# ============================================================

DEFAULT_DATA_DIR = Path(__file__).parent / "data/ruler_64k"


# ============================================================
# Chat Template Conversion
# ============================================================


def convert_llama_to_glm4_format(prompt: str) -> str:
    """
    Convert Llama 3 chat template format to GLM-4 format.

    Llama 3 format:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {assistant_prefix}

    GLM-4 format:
        [gMASK]<sop><|user|>
        {user_content}<|assistant|>
        {assistant_prefix}
    """
    # Split into user content and assistant prefix
    parts = prompt.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")

    # Extract user content (remove Llama header tokens)
    user_content = parts[0]
    user_content = user_content.replace("<|begin_of_text|>", "")
    user_content = user_content.replace("<|start_header_id|>user<|end_header_id|>", "")
    user_content = user_content.strip()

    # Extract assistant prefix (if exists)
    assistant_prefix = ""
    if len(parts) > 1:
        assistant_prefix = parts[1].replace("<|eot_id|>", "").strip()

    # Apply GLM-4 format
    glm_prompt = f"[gMASK]<sop><|user|>\n{user_content}<|assistant|>"
    if assistant_prefix:
        glm_prompt += f"\n{assistant_prefix}"

    return glm_prompt


def is_glm_model(model_path: str) -> bool:
    """Check if the model is a GLM model based on config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(config, "model_type", "") == "chatglm"


def convert_prompt_for_model(prompt: str, model_path: str) -> str:
    """Convert prompt format based on model type."""
    if is_glm_model(model_path):
        return convert_llama_to_glm4_format(prompt)
    return prompt  # Keep original format for Llama and other models


DEFAULT_MODEL = os.path.expanduser("~/models/Llama-3.1-8B-Instruct")
# Note: max_model_len must be > max_input_len to leave room for output tokens
# 64k benchmark has inputs up to 65536 tokens, so we need 65536 + 128 = 65664
DEFAULT_MAX_MODEL_LEN = 65664
DEFAULT_MAX_NEW_TOKENS = 16  # Sufficient for NIAH single-value answers

# Task categories for evaluation
NIAH_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
]
QA_TASKS = ["qa_1", "qa_2"]
RECALL_TASKS = ["cwe", "fwe", "vt"]

ALL_TASKS = NIAH_TASKS + QA_TASKS + RECALL_TASKS


# ============================================================
# Data Loading
# ============================================================


def load_samples(filepath: Path, indices: Optional[List[int]] = None) -> List[dict]:
    """Load samples from a JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if indices is None or i in indices:
                sample = json.loads(line)
                sample["_local_idx"] = i
                samples.append(sample)
    return samples


def count_samples(filepath: Path) -> int:
    """Count total samples in JSONL file."""
    with open(filepath) as f:
        return sum(1 for _ in f)


# ============================================================
# Evaluation Functions (Following RULER Official Metrics)
# Ref: https://github.com/NVIDIA/RULER/blob/main/scripts/eval/synthetic/constants.py
# ============================================================


def string_match_all(output_text: str, expected_list: List[str]) -> float:
    """
    RULER official metric for NIAH, VT, CWE, FWE tasks.

    Formula: sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)

    Returns recall score (0.0 to 1.0): fraction of expected values found in output.
    """
    output_clean = (
        output_text.replace("<|im_end|>", "").replace("\r", " ").replace("\n", " ")
    )
    output_lower = output_clean.lower()

    if not expected_list:
        return 1.0

    found = sum(
        1.0 if exp.strip().lower() in output_lower else 0.0 for exp in expected_list
    )
    return found / len(expected_list)


def string_match_part(output_text: str, expected_list: List[str]) -> float:
    """
    RULER official metric for QA tasks.

    Formula: max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])

    Returns 1.0 if ANY expected value is found, 0.0 otherwise.
    """
    output_clean = (
        output_text.replace("<|im_end|>", "").replace("\r", " ").replace("\n", " ")
    )
    output_lower = output_clean.lower()

    if not expected_list:
        return 1.0

    return max(
        1.0 if exp.strip().lower() in output_lower else 0.0 for exp in expected_list
    )


def evaluate_output(
    output_text: str, expected_outputs: List[str], task_name: str
) -> Tuple[bool, float]:
    """
    Evaluate model output using RULER official metrics.

    - QA tasks: string_match_part (any match = full score)
    - All other tasks: string_match_all (recall-based score)

    Returns (passed, score) where passed = score >= 0.5
    """
    if task_name in QA_TASKS:
        score = string_match_part(output_text, expected_outputs)
    else:
        # NIAH, VT, CWE, FWE all use string_match_all
        score = string_match_all(output_text, expected_outputs)

    passed = score >= 0.5  # Consider pass if score >= 50%
    return passed, score


# ============================================================
# Test Runner
# ============================================================


def run_task_test(
    llm: LLM,
    task_name: str,
    data_dir: Path,
    sample_indices: Optional[List[int]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    verbose: bool = True,
    llm_factory: Optional[callable] = None,
    fresh_llm: bool = False,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Run test for a single RULER task.

    Args:
        llm: LLM instance (ignored if fresh_llm=True)
        task_name: Name of the task to test
        data_dir: Path to data directory
        sample_indices: Optional list of specific sample indices to test
        max_new_tokens: Maximum tokens to generate
        verbose: Print detailed output
        llm_factory: Callable to create LLM instance (required if fresh_llm=True)
        fresh_llm: If True, reinitialize LLM for each sample (avoids state leakage)

    Returns dict with: task, correct, total, score, results
    """
    data_file = data_dir / task_name / "validation.jsonl"
    samples = load_samples(data_file, sample_indices)

    if verbose:
        mode_str = " [fresh-llm mode]" if fresh_llm else ""
        print(f"\n  Testing {task_name}: {len(samples)} samples{mode_str}")

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=max_new_tokens,
    )

    correct = 0
    total_score = 0.0
    results = []

    current_llm = llm

    for sample in samples:
        idx = sample.get("index", sample["_local_idx"])
        prompt = sample["input"]
        # Convert prompt format for GLM models
        if model_path:
            prompt = convert_prompt_for_model(prompt, model_path)
        expected = sample["outputs"]

        # Fresh LLM mode: reinitialize for each sample
        if fresh_llm:
            if llm_factory is None:
                raise ValueError("llm_factory required when fresh_llm=True")
            # Cleanup previous LLM
            if current_llm is not None:
                del current_llm
                gc.collect()
                torch.cuda.empty_cache()
            current_llm = llm_factory()

        # Generate
        outputs = current_llm.generate([prompt], sampling_params, use_tqdm=False)
        output_text = outputs[0]["text"]

        # Evaluate
        passed, score = evaluate_output(output_text, expected, task_name)
        if passed:
            correct += 1
        total_score += score

        results.append(
            {
                "index": idx,
                "expected": expected,
                "output": output_text[:200],
                "passed": passed,
                "score": score,
            }
        )

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            exp_preview = str(expected[0])[:30] if expected else "N/A"
            out_preview = output_text[:50].replace("\n", " ")
            print(
                f"    [{idx:3d}] {status} (score={score:.2f}) exp={exp_preview}... | out={out_preview}..."
            )

    # Cleanup last LLM instance in fresh mode
    if fresh_llm and current_llm is not None:
        del current_llm
        gc.collect()
        torch.cuda.empty_cache()

    avg_score = total_score / len(samples) if samples else 0.0

    return {
        "task": task_name,
        "correct": correct,
        "total": len(samples),
        "accuracy": correct / len(samples) if samples else 0.0,
        "avg_score": avg_score,
        "results": results,
    }


def run_ruler_benchmark(
    model_path: str,
    data_dir: Path,
    datasets: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    sample_indices: Optional[List[int]] = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_cpu_offload: bool = False,
    num_gpu_blocks: int = 4,
    block_size: int = 4096,
    num_kv_buffers: int = 4,
    gpu_utilization: float = 0.9,
    enforce_eager: bool = True,
    verbose: bool = True,
    fresh_llm: bool = False,
    json_output: bool = False,
    sparse_policy: Optional[str] = None,
    sparse_threshold: float = 0.9,
    sparse_samples: int = 128,
    sparse_block_size: int = 128,
    sparse_stride: int = 8,
    blasst_lambda: float = None,
    dtype: str = None,
) -> Dict:
    """
    Run RULER benchmark on multiple tasks.

    Args:
        model_path: Path to the model
        data_dir: Directory containing task subdirectories
        datasets: List of task names to test (None = all)
        num_samples: Number of samples per task (None = all)
        sample_indices: Specific sample indices to test (overrides num_samples)
        fresh_llm: If True, reinitialize LLM for each sample (avoids state leakage)
        json_output: If True, output JSON results at the end
        sparse_policy: Sparse attention policy (FULL, QUEST, MINFERENCE, XATTN)

    Returns:
        Dict with overall results and per-task results
    """
    # Determine tasks to run
    if datasets is None:
        tasks = [t for t in ALL_TASKS if (data_dir / t / "validation.jsonl").exists()]
    else:
        tasks = datasets

    # Sample indices: explicit list takes precedence over num_samples
    if sample_indices is not None:
        indices = sample_indices
    elif num_samples:
        indices = list(range(num_samples))
    else:
        indices = None

    samples_desc = (
        str(sample_indices)
        if sample_indices
        else (str(num_samples) if num_samples else "all")
    )

    if not json_output:
        print(f"\n{'=' * 60}")
        print("RULER Benchmark")
        print(f"{'=' * 60}")
        print(f"Model: {model_path}")
        print(f"Data dir: {data_dir}")
        print(f"Tasks: {len(tasks)}")
        print(f"Samples: {samples_desc}")
        print(f"CPU offload: {enable_cpu_offload}")
        print(f"Fresh LLM mode: {fresh_llm}")
        print(f"{'=' * 60}")

    # Enable DensityObserver for XAttention BSA
    if sparse_policy and sparse_policy.upper() == "XATTN_BSA":
        DensityObserver.enable()
        DensityObserver.complete_reset()
        # Set mode for correct density interpretation
        DensityObserver.set_mode("offload" if enable_cpu_offload else "gpu_only")
        if not json_output:
            mode_str = "offload" if enable_cpu_offload else "gpu_only"
            print(f"[DensityObserver] Enabled for XAttention BSA (mode: {mode_str})")

    # LLM initialization kwargs
    llm_kwargs = {
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_model_len,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_utilization,
        "kvcache_block_size": block_size,
        "enable_cpu_offload": enable_cpu_offload,
    }
    if dtype:
        llm_kwargs["dtype"] = dtype
    if enable_cpu_offload:
        llm_kwargs["num_gpu_blocks"] = num_gpu_blocks
        llm_kwargs["num_kv_buffers"] = num_kv_buffers
    if sparse_policy:
        from nanovllm.config import SparsePolicyType

        sparse_policy_type = SparsePolicyType[sparse_policy]
        llm_kwargs["sparse_policy"] = sparse_policy_type
        # XAttention BSA specific parameters
        if sparse_policy_type == SparsePolicyType.XATTN_BSA:
            llm_kwargs["sparse_threshold"] = sparse_threshold
            llm_kwargs["sparse_samples_per_chunk"] = sparse_samples
            llm_kwargs["sparse_stride"] = sparse_stride
        elif sparse_policy_type == SparsePolicyType.COMPASS:
            if blasst_lambda is not None:
                llm_kwargs["lambda_threshold"] = blasst_lambda
        elif sparse_policy_type == SparsePolicyType.BLASST:
            if blasst_lambda is not None:
                llm_kwargs["fixed_lambda"] = blasst_lambda

    # Factory function for fresh_llm mode
    def create_llm():
        return LLM(model_path, **llm_kwargs)

    # Initialize LLM (only once if not fresh_llm mode)
    llm = None
    if not fresh_llm:
        if not json_output:
            print("\nInitializing LLM...")
        llm = create_llm()

    # Run tests
    start_time = time.time()
    task_results = []

    for task_name in tasks:
        result = run_task_test(
            llm=llm,
            task_name=task_name,
            data_dir=data_dir,
            sample_indices=indices,
            max_new_tokens=max_new_tokens,
            verbose=verbose and not json_output,
            llm_factory=create_llm,
            fresh_llm=fresh_llm,
            model_path=model_path,
        )
        task_results.append(result)

        if verbose and not json_output:
            print(
                f"  -> {task_name}: {result['correct']}/{result['total']} "
                f"({result['accuracy'] * 100:.1f}%) avg_score={result['avg_score']:.3f}"
            )

    total_time = time.time() - start_time

    # Print ChunkedPrefillTimer summary (for COMPASS/BLASST timing analysis)
    from nanovllm.layers.attention import ChunkedPrefillTimer
    ChunkedPrefillTimer().print_final_summary()

    # Statistics for COMPASS
    if sparse_policy and sparse_policy.upper() == "COMPASS":
        if llm is not None and hasattr(llm, "model_runner") and hasattr(llm.model_runner, "kvcache_manager"):
            kv_manager = llm.model_runner.kvcache_manager
            policy = kv_manager.sparse_policy
            if type(policy).__name__ == "COMPASSPolicy":
                stats = policy.get_stats()
                print(f"\n{'=' * 60}")
                print("COMPASS Block Selection Statistics")
                print(f"{'=' * 60}")
                print(f"  Chunks: {stats['num_chunks']}")
                print(f"  Selected blocks: {stats['selected_blocks']}/{stats['total_blocks']}")
                print(f"  Selection rate: {stats['select_rate']:.3f}")
                print(f"  IO reduction: {stats['io_reduction'] * 100:.1f}%")

    # Cleanup (only if not fresh_llm mode, since fresh mode cleans up itself)
    if llm is not None:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate results
    total_correct = sum(r["correct"] for r in task_results)
    total_samples = sum(r["total"] for r in task_results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_score = (
        sum(r["avg_score"] for r in task_results) / len(task_results)
        if task_results
        else 0.0
    )

    # Collect failed samples
    failed_samples = {}
    for r in task_results:
        failed = [res["index"] for res in r["results"] if not res["passed"]]
        if failed:
            failed_samples[r["task"]] = failed

    # Print summary
    if not json_output:
        print(f"\n{'=' * 60}")
        print("RULER Benchmark Results")
        print(f"{'=' * 60}")
        print(f"\n{'Task':<20} {'Correct':<10} {'Accuracy':<12} {'Avg Score':<12}")
        print(f"{'-' * 54}")
        for r in task_results:
            print(
                f"{r['task']:<20} {r['correct']}/{r['total']:<7} {r['accuracy'] * 100:>6.1f}%      {r['avg_score']:.3f}"
            )
        print(f"{'-' * 54}")
        print(
            f"{'TOTAL':<20} {total_correct}/{total_samples:<7} {overall_accuracy * 100:>6.1f}%      {avg_score:.3f}"
        )
        print(f"\nTime: {total_time:.1f}s")

        # Print DensityObserver summary if enabled
        if (
            sparse_policy
            and sparse_policy.upper() == "XATTN_BSA"
            and DensityObserver.is_enabled()
        ):
            print(f"\n{'=' * 60}")
            print("Density Statistics (XAttention BSA)")
            print(f"{'=' * 60}")
            DensityObserver.print_summary()

        print(f"{'=' * 60}\n")

    results = {
        "total_correct": total_correct,
        "total_samples": total_samples,
        "overall_accuracy": overall_accuracy,
        "avg_score": avg_score,
        "time": total_time,
        "task_results": task_results,
        "failed_samples": failed_samples,
    }

    # JSON output
    if json_output:
        json_results = {
            "total_correct": total_correct,
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "avg_score": avg_score,
            "time": total_time,
            "tasks": {
                r["task"]: {
                    "correct": r["correct"],
                    "total": r["total"],
                    "accuracy": r["accuracy"],
                }
                for r in task_results
            },
            "failed_samples": failed_samples,
        }
        print(json.dumps(json_results, indent=2))

    return results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RULER benchmark comprehensive test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to test (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples per dataset (default: 0 = all)",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Comma-separated specific sample indices (e.g., 28,33,40)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Maximum model context length (default: {DEFAULT_MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--enable-offload", action="store_true", help="Enable CPU offload mode"
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=4,
        help="Number of GPU blocks for CPU offload (default: 4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="KV cache block size (default: 4096)",
    )
    parser.add_argument(
        "--num-kv-buffers",
        type=int,
        default=4,
        help="Number of KV buffers for ring buffer (default: 4)",
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--use-cuda-graph", action="store_true", help="Enable CUDA graph"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument(
        "--fresh-llm",
        action="store_true",
        help="Reinitialize LLM for each sample (avoids state leakage)",
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "--sparse-policy",
        type=str,
        default="",
        help="Sparse attention policy (FULL, QUEST, XATTN_BSA)",
    )
    # XAttention BSA specific parameters
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=0.9,
        help="XAttention BSA: cumulative attention threshold (0-1)",
    )
    parser.add_argument(
        "--sparse-samples",
        type=int,
        default=128,
        help="XAttention BSA: samples per chunk for estimation",
    )
    parser.add_argument(
        "--sparse-block-size",
        type=int,
        default=128,
        help="XAttention BSA: block size for estimation",
    )
    parser.add_argument(
        "--sparse-stride",
        type=int,
        default=8,
        help="XAttention BSA: stride for Q/K downsampling",
    )
    parser.add_argument(
        "--blasst-lambda",
        type=float,
        default=None,
        help="BLASST: fixed threshold lambda (overrides dynamic formula)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype (bfloat16, float16). Required for models with float32 default.",
    )

    args = parser.parse_args()

    # Parse datasets
    datasets = args.datasets.split(",") if args.datasets else None
    num_samples = args.num_samples if args.num_samples > 0 else None

    # Parse sample indices (takes precedence over num_samples)
    sample_indices = None
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]

    # Parse sparse policy
    sparse_policy_str = args.sparse_policy.upper() if args.sparse_policy else None

    results = run_ruler_benchmark(
        model_path=os.path.expanduser(args.model),
        data_dir=Path(args.data_dir),
        datasets=datasets,
        num_samples=num_samples,
        sample_indices=sample_indices,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        enable_cpu_offload=args.enable_offload,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        num_kv_buffers=args.num_kv_buffers,
        gpu_utilization=args.gpu_utilization,
        enforce_eager=not args.use_cuda_graph,
        verbose=not args.quiet,
        fresh_llm=args.fresh_llm,
        json_output=args.json_output,
        sparse_policy=sparse_policy_str,
        sparse_threshold=args.sparse_threshold,
        sparse_samples=args.sparse_samples,
        sparse_block_size=args.sparse_block_size,
        sparse_stride=args.sparse_stride,
        blasst_lambda=args.blasst_lambda,
        dtype=args.dtype,
    )

    # Exit code (skip for json output mode)
    if not args.json_output:
        if results["overall_accuracy"] >= 0.5:
            print("test_ruler: PASSED")
        else:
            print(
                f"test_ruler: FAILED (accuracy={results['overall_accuracy'] * 100:.1f}%)"
            )
            exit(1)
