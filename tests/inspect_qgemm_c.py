import sys
import os

# Ensure nanovllm is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm.ops.tvm_qgemm.qgemm import QGeMMLUTBitsCodegen


def inspect_c_code(M, N, K, bits=2):
    target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"
    cc_opts = ["-O3", "-march=native", "-mllvm", "-inline-threshold=10000"]

    codegen = QGeMMLUTBitsCodegen(
        dtype="int8",
        target=target,
        name="qgemm_inspect",
        bits=bits,
        num_threads=4,
        save_dir="build/inspect_qgemm",
        verify=False,
        tune=False,
        cc_opts=cc_opts,
    )

    header, code = codegen.compile(M, N, K, return_type="c")
    print("--- HEADER ---")
    print(header)
    print("--- CODE ---")
    # Print first 500 lines of code
    lines = code.split("\n")
    print("\n".join(lines[:500]))
    if len(lines) > 500:
        print("...")


if __name__ == "__main__":
    inspect_c_code(M=65536, N=1, K=128, bits=2)
