#!/usr/bin/env python3
# Copyright (c) 2024 QLutAttn (Adapted for nano-vllm).
# Licensed under the MIT License.

import subprocess
import shutil
from typing import Tuple
import tarfile
from io import BytesIO
import os
import urllib.request
import platform

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORM_LLVM_MAP = {
    # (system, processor): (llvm_version, file_suffix)
    ("Darwin", "aarch64"): ("17.0.6", "arm64-apple-darwin22.0.tar.xz"),
    ("Linux", "aarch64"): ("17.0.6", "aarch64-linux-gnu.tar.xz"),
    ("Linux", "x86_64"): ("17.0.6", "x86_64-linux-gnu-ubuntu-22.04.tar.xz"),
}

ARCH_MAP = {
    "arm64": "aarch64",
    "AMD64": "x86_64",
    "ARM64": "aarch64",
    "x86": "x86_64",
}


def get_system_info() -> Tuple[str, str]:
    """Get OS and processor architecture"""
    system = platform.system()
    processor = platform.machine()
    processor = ARCH_MAP.get(processor, processor)
    return system, processor


def is_win() -> bool:
    """Check if is windows or not"""
    return get_system_info()[0] == "Windows"


def download_and_extract_llvm(extract_path):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    """
    llvm_version, file_suffix = PLATFORM_LLVM_MAP[get_system_info()]
    base_url = (
        f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvm_version}"
    )
    file_name = f"clang+llvm-{llvm_version}-{file_suffix}"

    # Calculate the expected LLVM directory path
    llvm_dir = os.path.abspath(
        os.path.join(extract_path, file_name.replace(".tar.xz", "")))

    # Check if LLVM directory already exists
    if os.path.exists(llvm_dir):
        llvm_config_path = os.path.join(llvm_dir, "bin", "llvm-config")
        if os.path.exists(llvm_config_path):
            print(f"LLVM already exists at {llvm_dir}, skipping download.")
            return llvm_dir

    download_url = f"{base_url}/{file_name}"

    # Download the file
    print(f"Downloading {file_name} from {download_url}")
    with urllib.request.urlopen(download_url) as response:
        if response.status != 200:
            raise Exception(
                f"Download failed with status code {response.status}")
        file_content = response.read()
    
    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    print(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    print("Download and extraction completed successfully.")
    return os.path.abspath(
        os.path.join(extract_path, file_name.replace(".tar.xz", "")))


def update_submodules():
    """Updates git submodules."""
    try:
        print("Updating submodules...")
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT_DIR)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to update submodules") from error


def build_tvm(llvm_path, llvm_config_path):
    """Configures and builds TVM using the explicitly provided LLVM toolchain."""
    tvm_dir = os.path.join(ROOT_DIR, "3rdparty", "tvm")
    os.chdir(tvm_dir)
    if not os.path.exists("build"):
        os.makedirs("build")
    os.chdir("build")
    
    # Copy the config.cmake as a baseline
    if not os.path.exists("config.cmake"):
        shutil.copy("../cmake/config.cmake", "config.cmake")
    
    llvm_bin_dir = os.path.join(llvm_path, "bin")
    
    # Set LLVM path and strict C/CXX compilers in config.cmake
    with open("config.cmake", "a") as config_file:
        if is_win():
            import posixpath
            llvm_config_path = llvm_config_path.replace(os.sep, posixpath.sep)
        config_file.write(f"\nset(USE_LLVM {llvm_config_path})\n")
        # Ensure we strictly use the downloaded clang/clang++
        config_file.write(f"set(CMAKE_C_COMPILER {os.path.join(llvm_bin_dir, 'clang')})\n")
        config_file.write(f"set(CMAKE_CXX_COMPILER {os.path.join(llvm_bin_dir, 'clang++')})\n")

    # Override environment PATH to prioritize downloaded LLVM bin just for the subprocess
    env = os.environ.copy()
    env["PATH"] = f"{llvm_bin_dir}:{env.get('PATH', '')}"

    # Run CMake and make
    try:
        print(f"Configuring TVM with CMake (using LLVM at {llvm_path})...")
        subprocess.check_call(["cmake", ".."], env=env)
        if is_win():
            print("Building TVM...")
            subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], env=env)
        else:
            print("Building TVM (make -j128)...")
            subprocess.check_call(["make", "-j128"], env=env)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build TVM") from error
    finally:
        os.chdir(ROOT_DIR)


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then returns paths to it."""
    extract_path = os.path.join(ROOT_DIR, "build")
    extracted_dir = download_and_extract_llvm(extract_path)
    llvm_config_path = os.path.join(extracted_dir, "bin", "llvm-config")
    return extracted_dir, llvm_config_path


def main():
    if get_system_info() not in PLATFORM_LLVM_MAP:
        raise RuntimeError(
            "This setup script hasn't supported auto-build for your operating system or CPU."
        )

    # custom build tvm
    update_submodules()
    
    # Set up LLVM for TVM
    llvm_path, llvm_config = setup_llvm_for_tvm()
    llvm_bin_path = os.path.abspath(os.path.join(llvm_path, "bin"))
    
    # Build TVM
    build_tvm(llvm_path, llvm_config)

    # Set up basic paths
    project_root = os.path.abspath(ROOT_DIR)
    tvm_root = os.path.abspath(os.path.join(project_root, "3rdparty", "tvm"))
    tvm_lib_path = os.path.abspath(os.path.join(tvm_root, "build"))
    tvm_python_path = os.path.abspath(os.path.join(tvm_root, "python"))

    # Always generate environment file
    envs = """#!/bin/bash
# nano-vllm Environment Setup with TVM
# Auto-generated by scripts/setup_tvm.py - DO NOT EDIT MANUALLY

# Unset any existing environment variables to ensure clean state
unset TVM_ROOT
unset NANO_VLLM_ROOT

# Set project paths
export NANO_VLLM_ROOT="{project_root}"
export TVM_ROOT="{tvm_root}"

# Set build paths
export PATH="{llvm_bin_path}:$PATH"
export LD_LIBRARY_PATH="{tvm_lib_path}:$LD_LIBRARY_PATH"

# Set PYTHONPATH (includes TVM and nano-vllm's current dir)
export PYTHONPATH="{tvm_python_path}:${{NANO_VLLM_ROOT}}:$PYTHONPATH"

echo "nano-vllm environment with TVM loaded:"
echo "  NANO_VLLM_ROOT: $NANO_VLLM_ROOT"
echo "  TVM_ROOT: $TVM_ROOT"
""".format(
        project_root=project_root,
        tvm_root=tvm_root,
        llvm_bin_path=llvm_bin_path,
        tvm_python_path=tvm_python_path,
        tvm_lib_path=tvm_lib_path
    )

    # Ensure build directory exists
    build_dir = os.path.join(ROOT_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    env_file_path = os.path.abspath(os.path.join(build_dir, "nano-vllm-envs.sh"))
    with open(env_file_path, "w") as env_file:
        env_file.write(envs)
        
    print(f"\\nEnvironment file generated successfully.")
    print(f"Please set environment variables by running:")
    print(f"    source {os.path.relpath(env_file_path, ROOT_DIR)}")

if __name__ == "__main__":
    main()
