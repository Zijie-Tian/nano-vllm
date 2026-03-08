# Sparse Policy KV Cache Offload Control Guide

## Overview

Historically, the KV cache offload process (transferring newly generated KV blocks/chunks from the GPU memory to CPU) was hardcoded in the primary compute flow (inside `Attention._chunked_prefill_attention` and `ModelRunner.run_chunked_offload_decode`). The `OffloadEngine` was directly invoked by the framework, restricting the ability of specialized sparse attention policies to intercept, alter, or optimize this data transfer.

To enable more dynamic and context-aware CPU offloading architectures, control of these offloading steps has been refactored and delegated to the `SparsePolicy` itself.

## Refactored Interfaces

Two new methods have been added to the abstract base class `SparsePolicy` in `nanovllm/kvcache/sparse/policy.py`.

### 1. `offload_prefill_chunk`
```python
def offload_prefill_chunk(
    self,
    offload_engine: "OffloadEngine",
    layer_id: int,
    cpu_block_id: int,
    num_tokens: int,
    **kwargs,
) -> None:
```
**Trigger:**
Called per chunk by `Attention._chunked_prefill_attention` inside `nanovllm/layers/attention.py` immediately after the chunked prefill compute (`sparse_policy.compute_chunked_prefill`) finishes. 

**Default Behavior:**
The base implementation invokes `offload_engine.offload_prefill_buffer_async(layer_id, cpu_block_id, num_tokens)` which executes an asynchronous Device-to-Host (D2H) copy from the independent per-layer prefill buffer to the CPU.

### 2. `offload_decode_chunk`
```python
def offload_decode_chunk(
    self,
    offload_engine: "OffloadEngine",
    layer_id: int,
    cpu_block_id: int,
    **kwargs,
) -> None:
```
**Trigger:**
Called per block by `ModelRunner.run_chunked_offload_decode` inside `nanovllm/engine/model_runner.py`. During sequential decode, tokens are continuously appended to the `decode_k_buffer`. Once a logical block becomes completely full (`pos_in_block == self.block_size - 1`), the model runner loops over all layers and triggers this method.

**Default Behavior:**
The base implementation invokes `offload_engine.offload_decode_slot_layer(layer_id, cpu_block_id)`.

## Impact on Custom Policies

By default, existing policies (`FullAttentionPolicy`, `QuestPolicy`, `XAttentionBSAPolicy`, `COMPASSPolicy`, `BLASSTPolicy`) simply forward the call to their superclass, maintaining behavioral parity with the prior architecture. 

However, you can now override these methods when writing specialized sparse policies. This allows developers to:
- **Delay or Skip Offloads:** E.g., if a chunk/block is determined mathematically insignificant, the policy could decide not to offload it to CPU at all, saving PCI-e bandwidth.
- **Data Transformation Before Offload:** A policy could quantize, compress, or structurally alter the KV tensors in the buffer before committing the D2H transfer.
- **Schedule Alteration:** A policy could enforce custom stream synchronizations for offloading.

## Example Custom Implementation

```python
class CustomBandwidthSaverPolicy(SparsePolicy):
    
    def offload_prefill_chunk(
        self, offload_engine, layer_id, cpu_block_id, num_tokens, **kwargs
    ) -> None:
        # Example logic: Only offload if the chunk passes a certain entropy threshold
        if self._is_chunk_important(layer_id):
            super().offload_prefill_chunk(offload_engine, layer_id, cpu_block_id, num_tokens, **kwargs)
        else:
            # Skip the async D2H transfer entirely for this chunk
            pass
            
    def offload_decode_chunk(
        self, offload_engine, layer_id, cpu_block_id, **kwargs
    ) -> None:
        # Example logic: Always offload, but do custom metric tracking first
        self._record_offload_metrics(layer_id, cpu_block_id)
        super().offload_decode_chunk(offload_engine, layer_id, cpu_block_id, **kwargs)
```

## Related Files
* `nanovllm/kvcache/sparse/policy.py`
* `nanovllm/layers/attention.py`
* `nanovllm/engine/model_runner.py`
