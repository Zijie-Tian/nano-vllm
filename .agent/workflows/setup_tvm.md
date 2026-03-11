---
description: Environment framework construction workflow for TVM and project requirements
---
1. Check if the file `build/nano-vllm-envs.sh` exists in the filesystem.
// turbo
2. If `build/nano-vllm-envs.sh` does not exist, initialize TVM using the setup script.
`python3 scripts/setup_tvm.py`
// turbo
3. Before executing code or performing python tests, ensure `nano-vllm-envs.sh` is sourced.
`source build/nano-vllm-envs.sh`
