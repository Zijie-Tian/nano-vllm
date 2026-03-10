import torch
import time
from nanovllm.kvcache.quant.kcache_quant import pack_kvcache_tmac

def bench_packing_performance():
    device = "cuda:0"
    torch.cuda.set_device(device)
    
    # Model config (Llama-3-8B equivalent)
    n_head = 32
    head_dim = 128
    bits, g, bm, kfactor = 2, 4, 512, 16
    
    # Context lengths to test: 32k to 1M
    ctx_labels = ["32k", "64k", "128k", "256k", "512k", "1m"]
    ctx_values = [32768, 65536, 131072, 262144, 524288, 1048576]
    
    print(f"=== KV Cache Packing Performance (GPU: {torch.cuda.get_device_name()}) ===")
    print(f"{'Context':>8} | {'Pack (ms)':>12} | {'Throughput (GB/s)':>18}")
    print("-" * 60)
    
    for label, ctx_len in zip(ctx_labels, ctx_values):
        # Quantized KV (int8)
        kv_q = torch.randint(-2, 2, (1, n_head, ctx_len, head_dim), device=device, dtype=torch.int8)
        
        # Warmup
        for _ in range(3): pack_kvcache_tmac(kv_q, True, bits, g, bm, kfactor)
        torch.cuda.synchronize()
        
        iters = 50 if ctx_len < 262144 else 10
        if ctx_len >= 1048576: iters = 5 # 1M is slow
        
        t0 = time.time()
        for _ in range(iters):
            pack_kvcache_tmac(kv_q, True, bits, g, bm, kfactor)
        torch.cuda.synchronize()
        t_ms = (time.time() - t0) / iters * 1000
        
        # GB/s = (Size in GB) / (Time in seconds)
        # Input size: batch * n_head * ctx_len * head_dim * 1 byte
        size_gb = (1 * n_head * ctx_len * head_dim) / (1024**3)
        throughput = size_gb / (t_ms / 1000)
        
        print(f"{label:>8} | {t_ms:12.3f} | {throughput:18.2f}")
        del kv_q
        torch.cuda.empty_cache()

if __name__ == "__main__":
    bench_packing_performance()
