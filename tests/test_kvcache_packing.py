import torch
import numpy as np
import pytest
from nanovllm.ops.tvm_qgemm.utils.model_utils import preprocess_kvcache
from nanovllm.kvcache.quant.kcache_quant import pack_kvcache_tmac

def test_k_cache_packing_shape():
    """Verify K cache packing output shape matches expectations."""
    batch, n_head, chunk_size, head_dim = 1, 8, 64, 128
    bits = 2
    g = 4
    bm = 128 # M_expanded = 64 * 2 = 128
    kfactor = 16
    
    # Mock quantized KV cache (uint8)
    kv_cache = np.random.randint(0, 4, (batch, n_head, chunk_size, head_dim), dtype=np.uint8)
    scales = np.random.rand(batch, n_head, chunk_size, 1).astype(np.float32)
    
    # NumPy reference
    packed_kv_ref, _ = preprocess_kvcache(
        kv_cache, scales, is_key=True, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    # PyTorch implementation
    # Convert uint8 with bias 2^(bits-1) back to int8 for pack_kvcache_tmac
    bias = 1 << (bits - 1)
    kv_q = torch.from_numpy(kv_cache.astype(np.int16) - bias).to(torch.int8)
    
    packed_kv = pack_kvcache_tmac(
        kv_q, is_key=True, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    assert packed_kv.shape == packed_kv_ref.shape
    np.testing.assert_array_equal(packed_kv.numpy(), packed_kv_ref)

def test_v_cache_packing_shape():
    """Verify V cache packing output shape matches expectations."""
    batch, n_head, chunk_size, head_dim = 1, 8, 64, 128
    bits = 2
    g = 4
    bm = 256 # M_expanded = 128 * 2 = 256
    kfactor = 16
    
    # Mock quantized KV cache (uint8)
    kv_cache = np.random.randint(0, 4, (batch, n_head, chunk_size, head_dim), dtype=np.uint8)
    scales = np.random.rand(batch, n_head, 1, head_dim).astype(np.float32)
    
    # NumPy reference
    packed_kv_ref, _ = preprocess_kvcache(
        kv_cache, scales, is_key=False, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    # PyTorch implementation
    bias = 1 << (bits - 1)
    kv_q = torch.from_numpy(kv_cache.astype(np.int16) - bias).to(torch.int8)
    
    packed_kv = pack_kvcache_tmac(
        kv_q, is_key=False, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    assert packed_kv.shape == packed_kv_ref.shape
    np.testing.assert_array_equal(packed_kv.numpy(), packed_kv_ref)

def test_k_cache_packing_4bit():
    """Verify 4-bit K cache packing."""
    batch, n_head, chunk_size, head_dim = 1, 4, 128, 128
    bits = 4
    g = 4
    bm = 512 # M_expanded = 128 * 4 = 512
    kfactor = 16
    
    kv_cache = np.random.randint(0, 16, (batch, n_head, chunk_size, head_dim), dtype=np.uint8)
    scales = np.random.rand(batch, n_head, chunk_size, 1).astype(np.float32)
    
    packed_kv_ref, _ = preprocess_kvcache(
        kv_cache, scales, is_key=True, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    bias = 1 << (bits - 1)
    kv_q = torch.from_numpy(kv_cache.astype(np.int16) - bias).to(torch.int8)
    
    packed_kv = pack_kvcache_tmac(
        kv_q, is_key=True, bits=bits, g=g, bm=bm, kfactor=kfactor
    )
    
    np.testing.assert_array_equal(packed_kv.numpy(), packed_kv_ref)
