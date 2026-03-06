# GEMINI.md (Scripts)

## Profiling (nsys) Mandates

1.  **Use `scripts/profile_offload.sh`**: **MUST NOT** run `nsys profile` directly.
2.  **GPU Specification**: **MUST** use the `--gpu X` parameter of the script.
    - **CAUTION**: Externally setting `CUDA_VISIBLE_DEVICES` will be ignored or overwritten by the script.
3.  **Output**: Profiles are automatically written to `results/nsys/`.

## Profiling Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu` | 0 | GPU Index |
| `--policy` | full | `full` or `xattn` |
| `--ctx-len` | 64k | Context length (32k, 64k, 128k, 256k, 512k, 768k, 1m) |
| `--no-offload` | - | GPU-only mode (benchmark) |
| `--num-gpu-blocks` | 4 | Number of slots in ring buffer |

## Example Command

```bash
bash scripts/profile_offload.sh --gpu 4 --policy xattn --ctx-len 128k --model ~/models/GLM-4-9B-Chat-1M
```
