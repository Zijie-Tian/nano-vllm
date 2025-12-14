# Testing

## Chunked Attention Test

```bash
CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 64 2
# Args: num_gpu_blocks input_len output_len num_prefetch_blocks
```

## CPU Offload Testing

```bash
# Basic test with limited GPU blocks to trigger offload
CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 64 2

# Verify consistency (run multiple times, output should be identical)
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=4,5 python tests/test_chunked_attention.py 6 2048 32 2 2>&1 | tail -3
done
```
