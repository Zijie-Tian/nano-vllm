---
description: Setup TVM environment and workspace protocols
---
# 1. Environment & Setup Checklist

1. **Source the Environment Script**
   Before executing **any** code or scripts in this repository, you **MUST** configure the environment by sourcing `build/nano-vllm-envs.sh`. This script configures `PYTHONPATH=$(pwd):$PYTHONPATH` and sets up TVM paths.

2. **Pre-execution Check**
   Always check if `build/nano-vllm-envs.sh` exists.

3. **TVM Configuration Fallback**
   If the script does *not* exist, you must configure TVM first by running:
   ```bash
   python3 scripts/setup_tvm.py
   ```
   Wait for the build to complete, then source the script: `source build/nano-vllm-envs.sh`.

# 2. Planning Protocol
* Planning Files: Use `findings.md`, `task_plan.md`, and `progress.md` for complex tasks. These are excluded from git.
* Automatic Cleanup: At the beginning of every new task, you **MUST** automatically delete any existing `task_plan.md`, `findings.md`, and `progress.md` files to ensure a fresh state. This rule is particularly enforced for Gemini multi-turn continuity or starting new sessions.

# 3. Documentation Protocol
* Documentation Indexing: Whenever a new document is added to the `docs/` directory, its path and purpose **MUST** be immediately indexed in both `GEMINI.md`, `CLAUDE.md`, and `.agent/rules/1_project_overview.md`.
