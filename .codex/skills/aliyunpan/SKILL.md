---
name: aliyunpan
description: Download project assets from AliyunPan with the local `aliyunpan` CLI. Use when KVCache/RoPE or other repo data is missing locally and `.codex/rules/kvcache-rope-data.md` says to fetch it.
---

# AliyunPan Download

This skill provides the Codex-side equivalent of the AliyunPan download flow referenced by `.codex/rules/kvcache-rope-data.md`.

## First step
- Confirm the exact remote path and the exact local target directory.
- Check whether the target file already exists locally before downloading.

## Use the local CLI
This repository already has a local CLI available at `/usr/local/bin/aliyunpan`. Prefer the CLI over ad-hoc browser downloads.

## Required workflow
1. `mkdir -p <target_dir>`
2. Clear proxy variables before download:
   - `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`
3. Download with a direct target directory when possible:
   - `aliyunpan download --saveto <target_dir> <remote_path>`
4. If the CLI creates nested directories such as `{uid}/data/COMPASS/...`, move the requested files back into `<target_dir>` and clean up the empty tree.
5. Verify the final files exist and match the expected size/count before continuing.

## Safety notes
- Do not pre-download large trees unless the task truly needs them.
- Prefer downloading only the layers/files the task needs right now.
- Re-check file existence even if a previous session downloaded the data because `results/` is not guaranteed to persist across machines.

