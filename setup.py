import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="nano-vllm",
    version="0.2.0",
    author="Zijie Tian",
    description="A lightweight vLLM implementation with CUDA sgDMA support",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="nanovllm.comm._sgdma_cuda",
            sources=[
                "csrc/sgdma.cpp",
                "csrc/sgdma_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
            include_dirs=[
                os.path.join(ROOT, "csrc"),
            ],
        ),
        CppExtension(
            name="nanovllm.sparse._cpu_ops",
            sources=[
                "csrc/cpu_ops/qk_blockmask_torch.cpp",
                "csrc/cpu_ops/qk_blockmask_fp32.cpp",
                "csrc/cpu_ops/qk_blockmask_vnni.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "csrc/cpu_ops/include"),
            ],
            extra_compile_args=[
                "-O3", "-std=c++17", "-fopenmp",
                "-mavx512f", "-mavx512bw", "-mavx512vnni",
            ],
            extra_link_args=["-lgomp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10,<3.13",
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
        "transformers>=4.51.0",
        "flash-attn",
        "xxhash",
    ],
)

