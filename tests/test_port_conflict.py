"""Test for torch distributed port conflict fix.

This test verifies that:
1. Multiple independent processes can run simultaneously (dynamic port allocation)
2. Sequential LLM creation in same process works (proper cleanup)

Usage:
    # Test parallel processes (requires 2 GPUs)
    python tests/test_port_conflict.py --model ~/models/Qwen3-4B --gpus 4,5 --test parallel

    # Test sequential creation in same process
    CUDA_VISIBLE_DEVICES=4 python tests/test_port_conflict.py --model ~/models/Qwen3-4B --test sequential
"""

import argparse
import os
import subprocess
import sys
import time


def test_sequential_creation(model_path: str, enable_offload: bool = True):
    """Test creating multiple LLM instances sequentially in same process."""
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from nanovllm import LLM, SamplingParams

    print("=" * 60)
    print("Test: Sequential LLM Creation (same process)")
    print("=" * 60)

    for i in range(3):
        print(f"\n--- Creating LLM instance {i+1}/3 ---")

        llm_kwargs = {"enable_cpu_offload": enable_offload}
        if enable_offload:
            llm_kwargs["num_gpu_blocks"] = 2

        llm = LLM(model_path, **llm_kwargs)

        # Simple generation
        outputs = llm.generate(
            ["Hello, how are you?"],
            SamplingParams(max_tokens=20)
        )
        print(f"Output: {outputs[0]['text'][:50]}...")

        # Explicit cleanup
        llm.close()
        print(f"Instance {i+1} closed successfully")

    print("\n" + "=" * 60)
    print("PASSED: test_sequential_creation")
    print("=" * 60)


def test_context_manager(model_path: str, enable_offload: bool = True):
    """Test LLM with context manager."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from nanovllm import LLM, SamplingParams

    print("=" * 60)
    print("Test: Context Manager")
    print("=" * 60)

    for i in range(2):
        print(f"\n--- Context manager instance {i+1}/2 ---")

        llm_kwargs = {"enable_cpu_offload": enable_offload}
        if enable_offload:
            llm_kwargs["num_gpu_blocks"] = 2

        with LLM(model_path, **llm_kwargs) as llm:
            outputs = llm.generate(
                ["What is 2+2?"],
                SamplingParams(max_tokens=20)
            )
            print(f"Output: {outputs[0]['text'][:50]}...")

        print(f"Instance {i+1} auto-closed via context manager")

    print("\n" + "=" * 60)
    print("PASSED: test_context_manager")
    print("=" * 60)


def test_parallel_processes(model_path: str, gpus: str, enable_offload: bool = True):
    """Test running multiple nanovllm processes in parallel."""
    gpu_list = [int(g.strip()) for g in gpus.split(",")]
    if len(gpu_list) < 2:
        print("ERROR: Need at least 2 GPUs for parallel test")
        return False

    print("=" * 60)
    print(f"Test: Parallel Processes (GPUs: {gpu_list})")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Script to run in each subprocess
    script = f'''
import sys
sys.path.insert(0, "{project_root}")
import os
from nanovllm import LLM, SamplingParams

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"[GPU {{gpu}}] Starting LLM...")

llm_kwargs = {{"enable_cpu_offload": {enable_offload}}}
if {enable_offload}:
    llm_kwargs["num_gpu_blocks"] = 2

llm = LLM("{model_path}", **llm_kwargs)
print(f"[GPU {{gpu}}] LLM initialized, generating...")

outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=10))
print(f"[GPU {{gpu}}] Output: {{outputs[0]['text'][:30]}}...")

llm.close()
print(f"[GPU {{gpu}}] Done")
'''

    # Start processes on different GPUs
    procs = []
    for i, gpu in enumerate(gpu_list[:2]):  # Use first 2 GPUs
        print(f"\nStarting process on GPU {gpu}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        p = subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        procs.append((gpu, p))
        time.sleep(2)  # Stagger starts to see concurrent running

    # Wait and collect results
    all_passed = True
    for gpu, p in procs:
        stdout, _ = p.communicate(timeout=300)
        print(f"\n--- GPU {gpu} output ---")
        print(stdout)

        if p.returncode != 0:
            print(f"ERROR: GPU {gpu} process failed with code {p.returncode}")
            all_passed = False
        else:
            print(f"GPU {gpu} process completed successfully")

    print("\n" + "=" * 60)
    if all_passed:
        print("PASSED: test_parallel_processes")
    else:
        print("FAILED: test_parallel_processes")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test port conflict fix")
    parser.add_argument("--model", "-m", required=True, help="Path to model")
    parser.add_argument("--gpus", default="0,1", help="GPUs to use for parallel test (comma-separated)")
    parser.add_argument("--test", choices=["sequential", "context", "parallel", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--no-offload", action="store_true", help="Disable CPU offload")
    args = parser.parse_args()

    enable_offload = not args.no_offload
    model_path = os.path.expanduser(args.model)

    print(f"Model: {model_path}")
    print(f"CPU Offload: {enable_offload}")
    print(f"GPUs for parallel test: {args.gpus}")
    print()

    if args.test in ["sequential", "all"]:
        test_sequential_creation(model_path, enable_offload)
        print()

    if args.test in ["context", "all"]:
        test_context_manager(model_path, enable_offload)
        print()

    if args.test in ["parallel", "all"]:
        test_parallel_processes(model_path, args.gpus, enable_offload)


if __name__ == "__main__":
    main()
