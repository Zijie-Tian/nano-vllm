import torch
import pytest
from nanovllm.kvcache.quant import quantize_kcache_per_token, dequantize_kcache_per_token

def test_quantize_kcache_symmetric():
    """Test symmetric 2-bit per-token quantization."""
    # Create dummy K cache: [batch=1, n_heads=2, seq_len=4, head_dim=64]
    torch.manual_seed(42)
    k_cache = torch.randn(1, 2, 4, 64, dtype=torch.float16) * 10
    
    bits = 2
    k_q, scales, zero_points = quantize_kcache_per_token(k_cache, bits=bits, sym=True)
    
    assert k_q.shape == k_cache.shape
    assert k_q.dtype == torch.int8
    assert scales.shape == (1, 2, 4, 1)
    assert zero_points is None
    
    # Check bounds
    q_max = (1 << (bits - 1)) - 1
    assert k_q.max().item() <= q_max
    assert k_q.min().item() >= -q_max
    
    # Dequantize and check NMSE roughly
    k_dq = dequantize_kcache_per_token(k_q, scales, zero_points)
    assert k_dq.shape == k_cache.shape
    assert k_dq.dtype == torch.float16

def test_quantize_kcache_asymmetric():
    """Test asymmetric 2-bit per-token quantization."""
    torch.manual_seed(42)
    k_cache = torch.randn(1, 2, 4, 64, dtype=torch.float16) * 10
    # Add a large offset to test asymmetric behavior
    k_cache += 20.0
    
    bits = 2
    k_q, scales, zero_points = quantize_kcache_per_token(k_cache, bits=bits, sym=False)
    
    assert k_q.shape == k_cache.shape
    assert k_q.dtype == torch.int8
    assert scales.shape == (1, 2, 4, 1)
    assert zero_points.shape == (1, 2, 4, 1)
    
    q_min = -(1 << (bits - 1))
    q_max = (1 << (bits - 1)) - 1
    assert k_q.max().item() <= q_max
    assert k_q.min().item() >= q_min
    
    k_dq = dequantize_kcache_per_token(k_q, scales, zero_points)
    assert k_dq.shape == k_cache.shape
    assert k_dq.dtype == torch.float16
    
    # Just basic check that correlation is high
    corr = torch.corrcoef(torch.stack([k_cache.flatten().float(), k_dq.flatten().float()]))[0, 1]
    assert corr > 0.8, f"Correlation too low: {corr}"

if __name__ == "__main__":
    pytest.main([__file__])
