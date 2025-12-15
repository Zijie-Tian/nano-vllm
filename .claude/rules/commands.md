# Commands

## Installation

```bash
pip install -e .
```

## Running

```bash
# Run example
python example.py

# Run benchmarks
python bench.py                    # Standard benchmark
python bench_offload.py            # CPU offload benchmark
```

## Config Defaults

- `max_num_batched_tokens`: 16384
- `max_num_seqs`: 512
- `kvcache_block_size`: 4096
- `gpu_memory_utilization`: 0.9
- `enforce_eager`: False (enables CUDA graphs)
