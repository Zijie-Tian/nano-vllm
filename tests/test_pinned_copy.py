import torch
import torch.cuda.nvtx as nvtx
import time

def main():
    SLOTS = 4
    H = 8
    N = 4096
    D = 128
    dtype = torch.float16
    
    # EXACT shape of CPU staging buffer in offload_engine: [H, N, D]
    staging_k = torch.zeros((H, N, D), dtype=dtype, device="cpu", pin_memory=True)
    
    # EXACT shape of GPU k_cache in offload_engine: [SLOTS, H, N, D]
    k_cache_gpu = torch.zeros((SLOTS, H, N, D), dtype=dtype, device="cuda")
    
    stream = torch.cuda.Stream()
    
    # Warmup
    with torch.cuda.stream(stream):
        k_cache_gpu[0].copy_(staging_k, non_blocking=True)
    stream.synchronize()
    
    # Try a dynamic num_tokens to mirror actual execution
    num_tokens = 4096
    slot_idx = 0
    head_idx = 0
    
    time.sleep(1)
    
    # Scenario C: Exactly how load_staging_to_slot does it
    nvtx.range_push("Scenario_Exact_OffloadEngine_Slice")
    with torch.cuda.stream(stream):
        # We copy 4096 tokens for a SINGLE HEAD (head_idx=0).
        # This is exactly [4096, 128] elements -> 4096 * 128 * 2 = 1,048,576 bytes (1MB).
        # We want to see if nsys stats shows 1MB or 8MB.
        k_cache_gpu[slot_idx, head_idx, :num_tokens].copy_(staging_k[head_idx, :num_tokens], non_blocking=True)
    stream.synchronize()
    nvtx.range_pop()
    
    time.sleep(1)
    
    # Try an alternative: contiguous slices
    nvtx.range_push("Scenario_Contiguous_Slice")
    with torch.cuda.stream(stream):
        src = staging_k[head_idx, :num_tokens].contiguous()
        k_cache_gpu[slot_idx, head_idx, :num_tokens].copy_(src, non_blocking=True)
    stream.synchronize()
    nvtx.range_pop()

    print("Test completed. Check nsys sqlite for cuda_gpu_mem_size_sum.")

if __name__ == "__main__":
    main()
