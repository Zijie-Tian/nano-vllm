# 64K Prefill MLP Activation OOM Issue

## Problem Summary

When running RULER benchmark with 64K context length using CPU offload mode, OOM occurs during MLP forward pass in `run_layerwise_offload_prefill`. The KV cache is successfully offloaded to CPU, but MLP intermediate activations exceed available GPU memory.

## Environment

- GPU: RTX 3090 (24GB)
- Model: LLaMA 3.1 8B
- Sequence Length: 65536 tokens
- Mode: `enable_cpu_offload=True`, `num_gpu_blocks=2`

## Error Message

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.47 GiB.
GPU 0 has a total capacity of 23.57 GiB of which 2.66 GiB is free.
Including non-PyTorch memory, this process has 20.88 GiB memory in use.
Of the allocated memory 20.51 GiB is allocated by PyTorch, and 32.26 MiB
is reserved by PyTorch but unallocated.
```

## Stack Trace

```
File "nanovllm/engine/model_runner.py", line 843, in run_layerwise_offload_prefill
    hidden_states = layer.mlp(hidden_states)
  File "nanovllm/models/llama.py", line 103, in forward
    gate_up = self.gate_up_proj(x)
  File "nanovllm/layers/linear.py", line 73, in forward
    return F.linear(x, self.weight, self.bias)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.47 GiB.
```

## Root Cause Analysis

### Memory Breakdown

| Component | Calculation | Size |
|-----------|-------------|------|
| Model weights (BF16) | 8B params × 2 bytes | ~16 GB |
| GPU KV cache | 2 blocks × 1024 tokens × 8KB/token | ~16 MB |
| **Remaining for activations** | 24 - 16 - overhead | **~6-7 GB** |

### MLP Activation Memory (per layer)

For LLaMA 3.1 8B with `hidden_size=4096`, `intermediate_size=14336`:

| Tensor | Shape | Size (BF16) |
|--------|-------|-------------|
| MLP input | [65536, 4096] | 512 MB |
| gate_up output | [65536, 28672] | **3.47 GB** |
| down_proj input | [65536, 14336] | 1.75 GB |
| MLP output | [65536, 4096] | 512 MB |

**Peak MLP memory**: ~3.5-4 GB for intermediate tensors

### Why OOM Occurs

1. Model weights consume ~16 GB (loaded on GPU for layer-wise processing)
2. Available memory: ~7 GB
3. MLP `gate_up_proj` output: 3.47 GB
4. Additional tensors (input, gradients, etc.): ~1-2 GB
5. **Total required > Available** → OOM

## Code Location

The issue is in `nanovllm/engine/model_runner.py`:

```python
# Line 843 in run_layerwise_offload_prefill
hidden_states = layer.mlp(hidden_states)  # <-- OOM here
```

The entire sequence (65536 tokens) is passed through MLP in one shot.

## Current Configuration

From `model_wrappers.py` (RULER integration):

```python
llm_kwargs = {
    "max_model_len": max_model_len,           # 128 * 1024
    "max_num_batched_tokens": max_model_len,  # Same as max_model_len
    "enable_cpu_offload": True,
    "num_gpu_blocks": 2,
    ...
}
```

Setting `max_num_batched_tokens = max_model_len` causes nanovllm to process all tokens at once.

## Potential Solutions

### Option 1: Chunked MLP Processing

Modify `run_layerwise_offload_prefill` to process MLP in chunks:

```python
# Instead of:
hidden_states = layer.mlp(hidden_states)

# Do:
chunk_size = 8192  # Process 8K tokens at a time
chunks = hidden_states.split(chunk_size, dim=0)
outputs = []
for chunk in chunks:
    outputs.append(layer.mlp(chunk))
hidden_states = torch.cat(outputs, dim=0)
```

### Option 2: Activation Checkpointing

Use gradient checkpointing to recompute activations instead of storing them:

```python
from torch.utils.checkpoint import checkpoint
hidden_states = checkpoint(layer.mlp, hidden_states, use_reentrant=False)
```

### Option 3: Reduce Chunk Size via Config

Add a new config parameter `prefill_chunk_size` to control how many tokens are processed per forward pass.

## Memory Estimation Formula

For a given sequence length `S` and model config:

```
MLP_peak_memory = S × intermediate_size × 2 × 2 bytes
                = S × 14336 × 4 bytes

For S = 65536:
MLP_peak = 65536 × 14336 × 4 = 3.76 GB
```

Maximum safe sequence length for RTX 3090 (24GB):
```
S_max = available_memory / (intermediate_size × 4)
      = 6GB / (14336 × 4)
      ≈ 100K tokens (theoretical)
      ≈ 8-16K tokens (practical, with safety margin)
```

## Reproduction Steps

```bash
cd /home/zijie/Code/COMPASS/eval/RULER/scripts

# Set SEQ_LENGTHS to 65536 in config_models.sh
# Then run:
./run.sh llama3.1-8b-nanovllm synthetic --metric full --task niah_single_1
```

## Related Files

- `nanovllm/engine/model_runner.py`: `run_layerwise_offload_prefill()` (line 751+)
- `nanovllm/models/llama.py`: `LlamaMLP.forward()` (line 103)
- `nanovllm/config.py`: Config parameters
- RULER integration: `eval/RULER/scripts/pred/model_wrappers.py`
