"""
Test BLASST fused kernels for correctness.

Compares against reference implementations (flash_attn and existing merge).
"""

import torch
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

from nanovllm.ops.blasst_fused import blasst_chunk_attention, merge_attn_outputs

# ============================================================
# Test Configuration
# ============================================================

BATCH = 1
NUM_HEADS = 4
Q_LEN = 256
KV_LEN = 1024
HEAD_DIM = 64

SOFTMAX_SCALE = 1.0 / (HEAD_DIM ** 0.5)

# ============================================================
# Helper: Reference Flash Attention
# ============================================================

def flash_attn_reference(q, k, v, causal=False):
    """Simple PyTorch reference implementation."""
    # q, k, v: [batch, heads, seq, dim]
    q_len = q.shape[2]
    kv_len = k.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * SOFTMAX_SCALE

    if causal:
        mask = torch.triu(torch.ones(q_len, kv_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    # Compute LSE: max + log(sum(exp(score - max)))
    max_score = scores.max(dim=-1).values
    lse = max_score + torch.log(torch.sum(torch.exp(scores - max_score.unsqueeze(-1)), dim=-1))

    return out, lse


def merge_reference(o1, lse1, o2, lse2):
    """Reference merge implementation."""
    # o1, o2: [batch, heads, seq, head_dim]
    # lse1, lse2: [batch*heads, seq]
    batch, heads, seq, head_dim = o1.shape
    bh = batch * heads

    # Reshape LSE to match output shape
    lse1 = lse1.view(batch, heads, seq)
    lse2 = lse2.view(batch, heads, seq)

    # Compute in log space
    max_lse = torch.maximum(lse1, lse2)  # [batch, heads, seq]
    exp1 = torch.exp(lse1 - max_lse)     # [batch, heads, seq]
    exp2 = torch.exp(lse2 - max_lse)
    sum_exp = exp1 + exp2

    # Weighted average of outputs
    out = (o1 * exp1.unsqueeze(-1) + o2 * exp2.unsqueeze(-1)) / sum_exp.unsqueeze(-1)
    lse_out = max_lse + torch.log(sum_exp)  # [batch, heads, seq]
    lse_out = lse_out.view(bh, seq)         # [batch*heads, seq]

    return out, lse_out


# ============================================================
# Test 1: Chunk Attention
# ============================================================

def test_chunk_attention():
    """Test blasst_chunk_attention against reference."""
    print("=" * 60)
    print("Test 1: Chunk Attention")
    print("=" * 60)

    # Create input
    torch.manual_seed(42)
    q = torch.randn(BATCH, NUM_HEADS, Q_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, NUM_HEADS, KV_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, NUM_HEADS, KV_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)

    # Reference
    ref_out, ref_lse = flash_attn_reference(q.float(), k.float(), v.float())
    ref_out = ref_out.half()

    # Triton kernel
    tri_out, tri_lse = blasst_chunk_attention(q, k, v, SOFTMAX_SCALE, causal=False)

    # Check output
    out_diff = (ref_out - tri_out).abs().max().item()
    lse_diff = (ref_lse - tri_lse).abs().max().item()

    print(f"Output max diff: {out_diff:.6f}")
    print(f"LSE max diff: {lse_diff:.6f}")

    # Tolerance for FP16 (allow larger diff due to numerical precision)
    assert out_diff < 0.5, f"Output mismatch: {out_diff}"
    assert lse_diff < 0.5, f"LSE mismatch: {lse_diff}"

    print("✓ test_chunk_attention: PASSED\n")


# ============================================================
# Test 2: Chunk Attention Causal
# ============================================================

def test_chunk_attention_small():
    """Test with small block size (128x128)."""
    print("=" * 60)
    print("Test 2: Chunk Attention (128x128)")
    print("=" * 60)

    torch.manual_seed(42)
    q_len = 128
    kv_len = 128
    q = torch.randn(BATCH, NUM_HEADS, q_len, HEAD_DIM, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, NUM_HEADS, kv_len, HEAD_DIM, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, NUM_HEADS, kv_len, HEAD_DIM, device='cuda', dtype=torch.float16)

    # Reference
    ref_out, ref_lse = flash_attn_reference(q.float(), k.float(), v.float())
    ref_out = ref_out.half()

    # Triton kernel
    tri_out, tri_lse = blasst_chunk_attention(q, k, v, SOFTMAX_SCALE, causal=False)

    out_diff = (ref_out - tri_out).abs().max().item()
    lse_diff = (ref_lse - tri_lse).abs().max().item()

    print(f"Output max diff: {out_diff:.6f}")
    print(f"LSE max diff: {lse_diff:.6f}")

    # Tolerance for FP16 (allow larger diff due to numerical precision)
    assert out_diff < 0.5, f"Output mismatch: {out_diff}"
    assert lse_diff < 0.5, f"LSE mismatch: {lse_diff}"

    print("✓ test_chunk_attention_small: PASSED\n")


# ============================================================
# Test 3: Merge Outputs
# ============================================================

def test_merge_outputs():
    """Test merge_attn_outputs against reference."""
    print("=" * 60)
    print("Test 3: Merge Outputs")
    print("=" * 60)

    torch.manual_seed(42)
    o1 = torch.randn(BATCH, NUM_HEADS, Q_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    o2 = torch.randn(BATCH, NUM_HEADS, Q_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    lse1 = torch.randn(BATCH * NUM_HEADS, Q_LEN, device='cuda', dtype=torch.float32)
    lse2 = torch.randn(BATCH * NUM_HEADS, Q_LEN, device='cuda', dtype=torch.float32)

    # Reference
    ref_out, ref_lse = merge_reference(o1, lse1, o2, lse2)

    # Triton kernel
    tri_out, tri_lse = merge_attn_outputs(o1, lse1, o2, lse2)

    out_diff = (ref_out - tri_out).abs().max().item()
    lse_diff = (ref_lse - tri_lse).abs().max().item()

    print(f"Output max diff: {out_diff:.6f}")
    print(f"LSE max diff: {lse_diff:.6f}")

    assert out_diff < 0.01, f"Output mismatch: {out_diff}"
    assert lse_diff < 0.01, f"LSE mismatch: {lse_diff}"

    print("✓ test_merge_outputs: PASSED\n")


# ============================================================
# Test 4: End-to-End BLASST Pattern
# ============================================================

def test_end_to_end_blasst():
    """Test full BLASST pattern: compute attention and merge."""
    print("=" * 60)
    print("Test 4: End-to-End BLASST Pattern")
    print("=" * 60)

    torch.manual_seed(42)

    # Simulate: query sub-chunk against multiple KV sub-blocks
    q = torch.randn(BATCH, NUM_HEADS, 128, HEAD_DIM, device='cuda', dtype=torch.float16)

    # Two KV sub-blocks
    k1 = torch.randn(BATCH, NUM_HEADS, 128, HEAD_DIM, device='cuda', dtype=torch.float16)
    v1 = torch.randn(BATCH, NUM_HEADS, 128, HEAD_DIM, device='cuda', dtype=torch.float16)
    k2 = torch.randn(BATCH, NUM_HEADS, 128, HEAD_DIM, device='cuda', dtype=torch.float16)
    v2 = torch.randn(BATCH, NUM_HEADS, 128, HEAD_DIM, device='cuda', dtype=torch.float16)

    # Compute attention for each sub-block
    o1, lse1 = blasst_chunk_attention(q, k1, v1, SOFTMAX_SCALE, causal=False)
    o2, lse2 = blasst_chunk_attention(q, k2, v2, SOFTMAX_SCALE, causal=False)

    # Merge results
    merged_o, merged_lse = merge_attn_outputs(o1, lse1, o2, lse2)

    # Reference: full attention against concatenated KV
    k_full = torch.cat([k1, k2], dim=2)
    v_full = torch.cat([v1, v2], dim=2)
    ref_out, ref_lse = flash_attn_reference(q.float(), k_full.float(), v_full.float())
    ref_out = ref_out.half()

    out_diff = (ref_out - merged_o).abs().max().item()
    lse_diff = (ref_lse - merged_lse).abs().max().item()

    print(f"Merged output max diff: {out_diff:.6f}")
    print(f"Merged LSE max diff: {lse_diff:.6f}")

    assert out_diff < 0.1, f"Output mismatch: {out_diff}"
    assert lse_diff < 0.5, f"LSE mismatch: {lse_diff}"

    print("✓ test_end_to_end_blasst: PASSED\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\nBLASST Fused Kernels Test Suite")
    print("=" * 60)

    test_chunk_attention()
    test_chunk_attention_small()
    test_merge_outputs()
    test_end_to_end_blasst()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
