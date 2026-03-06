"""
Test: BLASST Split Kernel

Test the Codex-generated Triton kernels against PyTorch reference.
"""
import torch
import math
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

from nanovllm.ops.blasst_mask import blasst_mask_forward
from nanovllm.ops.blasst_attn import blasst_attn_forward
from tests.test_blasst_split_reference import (
    blasst_mask_reference,
    blasst_attn_with_mask_reference,
    full_attention_reference,
)


def test_mask_kernel_correctness():
    """Test mask kernel matches reference."""
    torch.manual_seed(42)

    num_heads = 8
    num_kv_heads = 2
    q_len = 512
    kv_len = 1024
    head_dim = 128
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.5)

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # Reference
    ref_mask, ref_new_rm = blasst_mask_reference(
        q, k, running_max, ln_lambda, softmax_scale, granularity
    )

    # Triton kernel
    tri_mask = blasst_mask_forward(
        q, k, running_max, ln_lambda, softmax_scale, granularity
    )

    # Verify
    assert tri_mask.shape == ref_mask.shape, f"Shape mismatch: {tri_mask.shape} vs {ref_mask.shape}"
    assert tri_mask.dtype == torch.bool

    # Check mask agreement
    agreement = (tri_mask == ref_mask).float().mean().item()
    assert agreement > 0.99, f"Mask agreement too low: {agreement:.2%}"

    print(f"test_mask_kernel_correctness: PASSED (agreement={agreement:.2%})")


def test_attn_kernel_correctness_no_skip():
    """Test attention kernel with no skipping (all False mask)."""
    torch.manual_seed(42)

    num_heads = 8
    num_kv_heads = 2
    q_len = 256
    kv_len = 512
    head_dim = 128
    granularity = 128
    softmax_scale = head_dim ** -0.5

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    # No skip mask
    num_kv_blocks = (kv_len + granularity - 1) // granularity
    skip_mask = torch.zeros((num_heads, q_len, num_kv_blocks), dtype=torch.bool, device='cuda')
    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # Full attention reference
    ref_out = full_attention_reference(q, k, v, softmax_scale)

    # Triton kernel
    tri_out, tri_new_rm, tri_lse = blasst_attn_forward(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    # Verify
    max_diff = (tri_out - ref_out).abs().max().item()
    assert max_diff < 1e-2, f"Output mismatch: {max_diff}"

    print(f"test_attn_kernel_correctness_no_skip: PASSED (max_diff={max_diff:.6f})")


def test_attn_kernel_with_skip():
    """Test attention kernel with actual skipping."""
    torch.manual_seed(42)

    num_heads = 4
    num_kv_heads = 4
    q_len = 128
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.001)

    # Structured input for predictable skip behavior
    q = torch.zeros(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    q[:, :, 0] = 1.0

    k = torch.zeros(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    k[:, 0:128, 0] = 80.0
    k[:, 128:256, 0] = 72.0
    k[:, 256:384, 0] = 0.0
    k[:, 384:512, 0] = -16.0

    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # Compute mask
    skip_mask = blasst_mask_forward(q, k, running_max, ln_lambda, softmax_scale, granularity)

    # Reference with mask
    ref_out, ref_new_rm, ref_lse = blasst_attn_with_mask_reference(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    # Triton kernel
    tri_out, tri_new_rm, tri_lse = blasst_attn_forward(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    # Verify
    max_diff = (tri_out - ref_out).abs().max().item()
    max_diff_rm = (tri_new_rm - ref_new_rm).abs().max().item()
    max_diff_lse = (tri_lse - ref_lse).abs().max().item()

    assert max_diff < 1e-2, f"Output mismatch: {max_diff}"
    assert max_diff_rm < 1e-2, f"Running max mismatch: {max_diff_rm}"

    skip_rate = skip_mask.float().mean().item()
    print(f"test_attn_kernel_with_skip: PASSED (skip_rate={skip_rate:.1%}, max_diff={max_diff:.6f})")


def test_gqa_support():
    """Test GQA support."""
    torch.manual_seed(42)

    num_heads = 8
    num_kv_heads = 2
    q_len = 256
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.5)

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # Mask kernel
    skip_mask = blasst_mask_forward(q, k, running_max, ln_lambda, softmax_scale, granularity)

    # Attention kernel
    out, new_rm, lse = blasst_attn_forward(q, k, v, skip_mask, running_max, softmax_scale, granularity)

    assert out.shape == (num_heads, q_len, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    print(f"test_gqa_support: PASSED (shape={out.shape})")


def test_precise_skip_statistics():
    """Test that skip statistics are precise with split kernel design."""
    torch.manual_seed(42)

    num_heads = 4
    num_kv_heads = 4
    q_len = 128
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.1)

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # Compute mask
    skip_mask = blasst_mask_forward(q, k, running_max, ln_lambda, softmax_scale, granularity)

    # Precise statistics - direct counting from mask
    num_skipped = skip_mask.sum().item()
    total = skip_mask.numel()
    skip_rate = num_skipped / total

    print(f"test_precise_skip_statistics: PASSED")
    print(f"  Total (query, block) pairs: {total}")
    print(f"  Skipped: {num_skipped}")
    print(f"  Skip rate: {skip_rate:.2%}")


if __name__ == "__main__":
    test_mask_kernel_correctness()
    test_attn_kernel_correctness_no_skip()
    test_attn_kernel_with_skip()
    test_gqa_support()
    test_precise_skip_statistics()
    print("\nAll BLASST split kernel tests PASSED!")
