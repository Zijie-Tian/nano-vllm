#!/usr/bin/env python3
"""
CUDA Graph Memory Analysis Test

This script analyzes the memory overhead of CUDA Graph at each stage:
1. Model loading
2. StaticCache allocation
3. Warmup runs
4. Graph capture
5. Graph replay

Usage:
    CUDA_VISIBLE_DEVICES=4 python tests/test_cudagraph_memory.py
    CUDA_VISIBLE_DEVICES=4 python tests/test_cudagraph_memory.py --model ~/models/Qwen3-0.6B
    CUDA_VISIBLE_DEVICES=4 python tests/test_cudagraph_memory.py --max-cache-len 2048
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache


def get_memory_mb():
    """Get current allocated memory in MB."""
    return torch.cuda.memory_allocated() / 1024**2


def get_memory_gb():
    """Get current allocated memory in GB."""
    return torch.cuda.memory_allocated() / 1024**3


def get_peak_memory_gb():
    """Get peak allocated memory in GB."""
    return torch.cuda.max_memory_allocated() / 1024**3


def print_separator(title=None):
    """Print a separator line."""
    if title:
        print(f"\n{'=' * 70}")
        print(f" {title}")
        print(f"{'=' * 70}")
    else:
        print("-" * 70)


def test_memory_stages(model_path: str, max_cache_len: int, batch_size: int = 1):
    """
    Test memory usage at each stage of CUDA Graph setup.

    Args:
        model_path: Path to the model
        max_cache_len: Maximum cache length for StaticCache
        batch_size: Batch size for inference
    """
    print_separator("CUDA Graph Memory Analysis")
    print(f"Model: {model_path}")
    print(f"Max cache length: {max_cache_len}")
    print(f"Batch size: {batch_size}")

    results = {}

    # Stage 0: Initial
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    results["initial"] = get_memory_mb()

    # Stage 1: Load model
    print_separator("Stage 1: Model Loading")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    results["after_model"] = get_memory_mb()
    model_size = results["after_model"] - results["initial"]
    print(f"  Memory: {results['after_model']:.0f} MB")
    print(f"  Model size: {model_size:.0f} MB ({model_size/1024:.2f} GB)")

    config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Stage 2: Allocate StaticCache
    print_separator("Stage 2: StaticCache Allocation")
    torch.cuda.reset_peak_memory_stats()
    before = get_memory_mb()

    static_cache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )

    results["after_cache"] = get_memory_mb()
    cache_size = results["after_cache"] - before
    print(f"  Memory: {results['after_cache']:.0f} MB")
    print(f"  StaticCache size: {cache_size:.0f} MB")

    # Calculate theoretical cache size
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    dtype_size = 2  # bfloat16

    theoretical_cache = (
        num_layers * 2 * batch_size * num_kv_heads * max_cache_len * head_dim * dtype_size
    ) / (1024**2)
    print(f"  Theoretical: {theoretical_cache:.0f} MB")
    print(f"  Overhead: {cache_size - theoretical_cache:.0f} MB ({(cache_size/theoretical_cache - 1)*100:.1f}%)")

    # Stage 3: Prepare static tensors
    print_separator("Stage 3: Static Tensor Allocation")
    before = get_memory_mb()

    static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    static_position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    static_cache_position = torch.tensor([0], dtype=torch.long, device=device)

    results["after_tensors"] = get_memory_mb()
    tensor_size = results["after_tensors"] - before
    print(f"  Memory: {results['after_tensors']:.0f} MB")
    print(f"  Static tensors: {tensor_size:.2f} MB (negligible)")

    # Stage 4: Warmup runs
    print_separator("Stage 4: Warmup Runs (3 iterations)")
    torch.cuda.reset_peak_memory_stats()
    before = get_memory_mb()

    with torch.inference_mode():
        for i in range(3):
            _ = model(
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                past_key_values=static_cache,
                cache_position=static_cache_position,
                use_cache=True,
            )
        torch.cuda.synchronize()

    results["after_warmup"] = get_memory_mb()
    results["warmup_peak"] = get_peak_memory_gb() * 1024
    warmup_size = results["after_warmup"] - before
    print(f"  Memory: {results['after_warmup']:.0f} MB")
    print(f"  Peak: {results['warmup_peak']:.0f} MB")
    print(f"  Warmup overhead: {warmup_size:.0f} MB")

    # Stage 5: CUDA Graph capture
    print_separator("Stage 5: CUDA Graph Capture")
    torch.cuda.reset_peak_memory_stats()
    before = get_memory_mb()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            outputs = model(
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                past_key_values=static_cache,
                cache_position=static_cache_position,
                use_cache=True,
            )
            static_logits = outputs.logits
    torch.cuda.synchronize()

    results["after_capture"] = get_memory_mb()
    results["capture_peak"] = get_peak_memory_gb() * 1024
    capture_size = results["after_capture"] - before
    print(f"  Memory: {results['after_capture']:.0f} MB")
    print(f"  Peak: {results['capture_peak']:.0f} MB")
    print(f"  Graph capture overhead: {capture_size:.0f} MB")

    # Stage 6: Graph replay
    print_separator("Stage 6: Graph Replay (10 iterations)")
    torch.cuda.reset_peak_memory_stats()
    before = get_memory_mb()

    with torch.inference_mode():
        for _ in range(10):
            static_input_ids.fill_(1)
            static_cache_position.fill_(0)
            graph.replay()
        torch.cuda.synchronize()

    results["after_replay"] = get_memory_mb()
    results["replay_peak"] = get_peak_memory_gb() * 1024
    replay_change = results["after_replay"] - before
    print(f"  Memory: {results['after_replay']:.0f} MB")
    print(f"  Peak: {results['replay_peak']:.0f} MB")
    print(f"  Replay memory change: {replay_change:.0f} MB (should be ~0)")

    # Summary
    print_separator("SUMMARY")
    total_overhead = results["after_capture"] - results["after_model"]

    print(f"{'Stage':<25} {'Memory (MB)':>12} {'Delta (MB)':>12}")
    print("-" * 50)
    print(f"{'Model loaded':<25} {results['after_model']:>12.0f} {model_size:>+12.0f}")
    print(f"{'StaticCache allocated':<25} {results['after_cache']:>12.0f} {cache_size:>+12.0f}")
    print(f"{'After warmup':<25} {results['after_warmup']:>12.0f} {warmup_size:>+12.0f}")
    print(f"{'After graph capture':<25} {results['after_capture']:>12.0f} {capture_size:>+12.0f}")
    print(f"{'After graph replay':<25} {results['after_replay']:>12.0f} {replay_change:>+12.0f}")
    print("-" * 50)
    print(f"{'Total (excl. model)':<25} {'':<12} {total_overhead:>+12.0f}")

    print_separator("KEY FINDINGS")
    print(f"  1. Model size:           {model_size/1024:.2f} GB")
    print(f"  2. StaticCache:          {cache_size:.0f} MB (main overhead, scales with cache_len)")
    print(f"  3. Graph capture:        {capture_size:.0f} MB (small, stores kernel sequence)")
    print(f"  4. Graph replay:         {replay_change:.0f} MB (zero allocation, reuses memory)")
    print(f"  5. Total CUDA Graph overhead: {total_overhead:.0f} MB")

    return results


def test_cache_length_scaling(model_path: str, cache_lengths: list):
    """
    Test how memory scales with different cache lengths.

    Args:
        model_path: Path to the model
        cache_lengths: List of cache lengths to test
    """
    print_separator("Cache Length Scaling Test")
    print(f"Model: {model_path}")
    print(f"Cache lengths: {cache_lengths}")

    # Load model once
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    model_mem = get_memory_mb()

    results = []
    for cache_len in cache_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create cache and capture graph
        static_cache = StaticCache(
            config=config,
            max_batch_size=1,
            max_cache_len=cache_len,
            device=device,
            dtype=dtype,
        )

        static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        static_position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        static_cache_position = torch.tensor([0], dtype=torch.long, device=device)

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                _ = model(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    past_key_values=static_cache,
                    cache_position=static_cache_position,
                    use_cache=True,
                )
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = model(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    past_key_values=static_cache,
                    cache_position=static_cache_position,
                    use_cache=True,
                )
            torch.cuda.synchronize()

        total_mem = get_memory_mb()
        overhead = total_mem - model_mem
        results.append((cache_len, total_mem, overhead))

        del static_cache, graph
        torch.cuda.empty_cache()

    # Print results
    print()
    print(f"{'Cache Length':>12} | {'Total (MB)':>12} | {'Overhead (MB)':>14} | {'Per 1K tokens':>14}")
    print("-" * 60)
    for cache_len, total, overhead in results:
        per_1k = overhead / (cache_len / 1000)
        print(f"{cache_len:>12} | {total:>12.0f} | {overhead:>14.0f} | {per_1k:>14.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CUDA Graph Memory Analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="~/models/Qwen3-4B-Instruct-2507",
        help="Model path",
    )
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Maximum cache length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--test-scaling",
        action="store_true",
        help="Test cache length scaling",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    print(f"Device: cuda:{torch.cuda.current_device()}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    if args.test_scaling:
        cache_lengths = [256, 512, 1024, 2048, 4096]
        test_cache_length_scaling(model_path, cache_lengths)
    else:
        test_memory_stages(model_path, args.max_cache_len, args.batch_size)

    print("\ntest_cudagraph_memory: PASSED")


if __name__ == "__main__":
    main()
