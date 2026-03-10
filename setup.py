from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
                "csrc",
            ],
        )
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
