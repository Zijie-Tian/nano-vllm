import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

# Import sparse policy classes
from nanovllm.kvcache.sparse.quest import QuestPolicy, QuestConfig, BlockMetadataManager
from nanovllm.kvcache.sparse.hybrid import HybridPolicy
from nanovllm.kvcache.sparse.full_policy import FullAttentionPolicy


def bench_decode(llm, num_seqs, input_len, max_output_len):
    """Benchmark decode performance (original test)"""
    seed(0)
    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_output_len) for _ in range(num_seqs)]

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_output_tokens / t
    print(f"[Decode] Output: {total_output_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


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


def setup_quest_policy(llm, topk_blocks=8, threshold_blocks=4):
    """
    Setup Quest sparse policy for decode phase.

    Uses HybridPolicy: Full attention for prefill, Quest Top-K for decode.
    """
    import torch

    kvcache_manager = llm.model_runner.kvcache_manager
    offload_engine = kvcache_manager.offload_engine

    # Get model parameters from offload engine
    num_layers = offload_engine.num_layers
    num_kv_heads = offload_engine.num_kv_heads
    head_dim = offload_engine.head_dim
    num_cpu_blocks = kvcache_manager.num_cpu_blocks
    dtype = offload_engine.k_cache_cpu.dtype

    print(f"Setting up Quest policy:")
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  num_cpu_blocks={num_cpu_blocks}, dtype={dtype}")
    print(f"  topk_blocks={topk_blocks}, threshold_blocks={threshold_blocks}")

    # Create BlockMetadataManager for storing min/max keys
    metadata = BlockMetadataManager(
        num_blocks=num_cpu_blocks,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # Create Quest policy for decode
    quest_config = QuestConfig(
        topk_blocks=topk_blocks,
        threshold_blocks=threshold_blocks,
    )
    quest_policy = QuestPolicy(quest_config, metadata)

    # Create Hybrid policy: Full for prefill, Quest for decode
    hybrid_policy = HybridPolicy(
        prefill_policy=FullAttentionPolicy(),
        decode_policy=quest_policy,
    )

    # Set the policy
    kvcache_manager.set_sparse_policy(hybrid_policy)
    print(f"  Policy set: HybridPolicy(prefill=Full, decode=Quest)")

    return hybrid_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sparse", action="store_true", help="Disable sparse attention (baseline)")
    parser.add_argument("--topk", type=int, default=8, help="Top-K blocks for Quest")
    parser.add_argument("--input-len", type=int, default=128 * 1024, help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=128, help="Output length in tokens")
    args = parser.parse_args()

    path = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507/")
    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=256 * 1024,
        max_num_batched_tokens=256 * 1024,
        enable_cpu_offload=True,
        num_gpu_blocks=120,
        num_prefetch_blocks=4,
    )

    if not args.no_sparse:
        # Setup Quest policy for decode (Top-K blocks, apply when > 4 blocks)
        setup_quest_policy(llm, topk_blocks=args.topk, threshold_blocks=4)
        print(f"\n[Quest Sparse Attention] topk={args.topk}")
    else:
        print("\n[Full Attention] No sparse policy (baseline)")

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    print("=" * 60)
    print("Prefill Benchmark (CPU Offload)")
    print("=" * 60)
    bench_prefill(llm, num_seqs=1, input_len=args.input_len)

    print("=" * 60)
    print("Decode Benchmark (CPU Offload)")
    print("=" * 60)
    bench_decode(llm, num_seqs=1, input_len=args.input_len, max_output_len=args.output_len)


if __name__ == "__main__":
    main()
