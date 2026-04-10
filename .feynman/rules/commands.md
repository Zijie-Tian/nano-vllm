# Commands

## Running (with PYTHONPATH)

For multi-instance development, use PYTHONPATH instead of pip install:

```bash
# Run example
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python example.py

# Run benchmarks
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python bench.py
PYTHONPATH=/path/to/nano-vllm:$PYTHONPATH python bench_offload.py
```

## Config Defaults

- `max_num_batched_tokens`: 16384
- `max_num_seqs`: 512
- `kvcache_block_size`: 4096
- `gpu_memory_utilization`: 0.9
- `enforce_eager`: False (enables CUDA graphs)
