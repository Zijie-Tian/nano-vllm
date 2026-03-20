# COMPASS V2 Jagged Architecture - Memory Utilization Notes

## The Issue: Suboptimal GPU VRAM Allocation

While the V2 Jagged Buffer architecture successfully avoids OOM by allocating a single globally shared buffer instead of layer-wise buffers, its peak capacity allocation is still functionally inefficient.

Currently, we allocate `max_jagged_tokens = num_cpu_blocks * block_size * num_kv_heads`. This assumes the absolute worst-case scenario: that for a given layer, the sparse policy (COMPASS) might select **100% of the historical context**.

## Why this is inefficient
Since COMPASS is fundamentally a **sparse** attention policy, the actual number of tokens gathered and transferred per layer is capped by the `top_p` or `top_k` threshold. For instance, if `top_p=0.9` translates to an average sparsity of ~10-20%, the Jagged Staging Buffer is 80-90% empty during every single layer execution. 

Even though the buffer is only ~150MB for 32k context, scaling to 1M context requires ~4GB. If we only ever select at most 20% of the blocks, we are wasting 3.2GB of VRAM that could be used for batching or other components.

## Future Optimization Path
After establishing a fully asynchronous pipeline (multi-buffer scheduling) to hide PCIe overhead, we should revisit the buffer allocation strategy:
1. **Dynamic Capacity Bounds**: Instead of allocating `100%` of context capacity, allocate the jagged buffer based on a strict upper bound derived from the target sparsity limit (e.g., if we strictly enforce max 30% blocks selected, the buffer only needs to be 30% of the size).
2. **Fallback Mechanism**: Add a safeguard mechanism during gather to handle edge cases if the selected blocks exceed the predicted dynamic capacity bound.
