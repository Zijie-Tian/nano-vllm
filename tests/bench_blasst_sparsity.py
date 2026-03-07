import torch
import triton
import triton.language as tl
import time
import os
import math

# Mask Buffer Kernel: Each program writes its own mask to a dedicated buffer
@triton.jit
def _blasst_mask_kernel(
    Q, K, V, sm_scale, threshold_ln_lambda,
    Out, Lse, Mask, # Mask [grid_0, grid_1, num_blocks]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lsez, stride_lseh, stride_lsem,
    stride_mask_g0, stride_mask_g1, stride_mask_b,
    Z, H, H_KV, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    
    H_i64 = H
    off_z = off_hz // H_i64
    off_h = off_hz % H_i64
    H_KV_i64 = H_KV
    off_h_kv = off_h // (H_i64 // H_KV_i64)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h_kv * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h_kv * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    
    # Point to the start of the mask slice for this program
    mask_ptrs = Mask + pid_m * stride_mask_g0 + off_hz * stride_mask_g1
    
    q_mask = (offs_m[:, None] < N_CTX_Q)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    block_idx = 0
    for start_n in range(0, N_CTX_K, BLOCK_N):
        k_mask_1d = (offs_n + start_n) < N_CTX_K
        k = tl.load(k_ptrs, mask=k_mask_1d[:, None], other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(q_mask & k_mask_1d[None, :], qk, float("-inf"))
        m_local = tl.max(qk, axis=1)
        
        diff = m_local - m_i
        max_diff = tl.max(diff, axis=0)
        
        is_computed = max_diff >= threshold_ln_lambda
        
        # Write to mask buffer: 1 if computed, 0 if skipped
        tl.store(mask_ptrs + block_idx * stride_mask_b, is_computed.to(tl.int8))
        
        if is_computed:
            m_new = tl.maximum(m_i, m_local)
            p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
            scale_factor = tl.math.exp2((m_i - m_new) * 1.44269504)
            l_new = l_i * scale_factor + tl.sum(p, axis=1)
            v = tl.load(v_ptrs, mask=k_mask_1d[:, None], other=0.0)
            acc = acc * scale_factor[:, None]
            acc = acc + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            l_i = l_new
        
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        block_idx += 1
        
    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i) * 0.69314718
    
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)
    tl.store(lse_ptrs, lse, mask=(offs_m < N_CTX_Q))

def run_bench_mask(q, k, v, threshold, iters=100):
    batch, heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, q_len), device='cuda', dtype=torch.float32)
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(q_len, BLOCK_M), batch * heads)
    num_blocks = triton.cdiv(kv_len, BLOCK_N)
    
    # Pre-allocate mask buffer
    mask_buffer = torch.zeros((grid[0], grid[1], num_blocks), device='cuda', dtype=torch.int8)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _blasst_mask_kernel[grid](
            q, k, v, sm_scale, threshold, out, lse, mask_buffer,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            mask_buffer.stride(0), mask_buffer.stride(1), mask_buffer.stride(2),
            batch, heads, num_kv_heads, q_len, kv_len,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_dim, BLOCK_N=BLOCK_N,
            num_warps=8, num_stages=2
        )
    torch.cuda.synchronize()
    t = (time.time() - start) / iters
    
    # Compute density from mask: 1.0 means all computed, 0.0 means all skipped
    actual_density = mask_buffer.float().mean().item()
    
    print(f"Density: {actual_density*100:.2f}%")
    print(f"Time: {t*1000:.3f} ms")
    return actual_density

def load_real_data(ctx_len='128k'):
    path = f'tests/data/real_kvcache/{ctx_len}/layer_05.pt'
    if not os.path.exists(path):
        return None, None, None
    data = torch.load(path)
    q = data['post_rope_q'].unsqueeze(0).transpose(1, 2).cuda().contiguous()
    k = data['post_rope_k'].unsqueeze(0).transpose(1, 2).cuda().contiguous()
    v = data['v'].unsqueeze(0).transpose(1, 2).cuda().contiguous()
    q = q[:, :, :128, :].contiguous()
    return q, k, v

if __name__ == "__main__":
    for ctx in ['16k', '32k', '64k', '128k']:
        print(f"\n--- Testing Real Data: {ctx} ---")
        q, k, v = load_real_data(ctx)
        if q is not None:
            print(f"Threshold: -0.69 (lambda=0.5)")
            run_bench_mask(q, k, v, -0.69)
            print(f"Threshold: -2.3 (lambda=0.1)")
            run_bench_mask(q, k, v, -2.3)
