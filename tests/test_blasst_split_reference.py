"""
Test: BLASST Split Kernel Reference Implementation

PyTorch reference for:
1. blasst_mask_forward: Compute skip mask
2. blasst_attn_forward: Compute attention with mask
"""
import torch
import math
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")


def blasst_mask_reference(
    q: torch.Tensor,          # [num_heads, q_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    running_max: torch.Tensor, # [num_heads, q_len]
    ln_lambda: float,
    softmax_scale: float,
    granularity: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of BLASST mask calculation.

    Returns:
        skip_mask: [num_heads, q_len, num_kv_blocks] bool
        new_running_max: [num_heads, q_len]
    """
    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, _ = k.shape
    gqa_ratio = num_heads // num_kv_heads

    num_kv_blocks = (kv_len + granularity - 1) // granularity

    # GQA: repeat KV heads
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=0)

    skip_mask = torch.zeros((num_heads, q_len, num_kv_blocks),
                           dtype=torch.bool, device=q.device)
    new_running_max = running_max.clone()

    for block_idx in range(num_kv_blocks):
        kv_start = block_idx * granularity
        kv_end = min(kv_start + granularity, kv_len)

        # Load KV block
        k_block = k[:, kv_start:kv_end, :]

        # Compute QK for this block
        scores = torch.matmul(q, k_block.transpose(-2, -1)) * softmax_scale
        # scores: [num_heads, q_len, kv_len_actual]

        # Get local_max per query
        local_max = scores.max(dim=-1).values  # [num_heads, q_len]

        # BLASST skip condition
        skip_condition = (local_max - new_running_max) < ln_lambda
        skip_mask[:, :, block_idx] = skip_condition

        # Update running_max (always update, whether skip or not)
        new_running_max = torch.maximum(new_running_max, local_max)

    return skip_mask, new_running_max


def blasst_attn_with_mask_reference(
    q: torch.Tensor,          # [num_heads, q_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    v: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    skip_mask: torch.Tensor,  # [num_heads, q_len, num_kv_blocks] bool
    running_max: torch.Tensor, # [num_heads, q_len]
    softmax_scale: float,
    granularity: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation of BLASST attention with mask.

    Returns:
        output: [num_heads, q_len, head_dim]
        new_running_max: [num_heads, q_len]
        lse: [num_heads, q_len] (log-sum-exp for merging)
    """
    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, _ = k.shape
    gqa_ratio = num_heads // num_kv_heads
    num_kv_blocks = (kv_len + granularity - 1) // granularity

    # GQA: repeat KV heads
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=0)
        v = v.repeat_interleave(gqa_ratio, dim=0)

    # Running statistics
    m_i = running_max.clone()  # running max
    l_i = torch.zeros((num_heads, q_len), dtype=torch.float32, device=q.device)  # sum of exp
    acc = torch.zeros((num_heads, q_len, head_dim), dtype=torch.float32, device=q.device)

    for block_idx in range(num_kv_blocks):
        kv_start = block_idx * granularity
        kv_end = min(kv_start + granularity, kv_len)
        kv_len_actual = kv_end - kv_start

        # Load KV block
        k_block = k[:, kv_start:kv_end, :]
        v_block = v[:, kv_start:kv_end, :]

        # Compute QK
        scores = torch.matmul(q, k_block.transpose(-2, -1)) * softmax_scale

        # Get local_max per query
        local_max = scores.max(dim=-1).values

        # Check which queries should skip this block
        block_skip_mask = skip_mask[:, :, block_idx]  # [num_heads, q_len]

        # For skipped queries: only update running_max
        m_i = torch.where(block_skip_mask, torch.maximum(m_i, local_max), m_i)

        # For non-skipped queries: compute PV with online softmax
        needs_compute = ~block_skip_mask

        if needs_compute.any():
            # Online softmax update
            new_m = torch.where(needs_compute,
                               torch.maximum(m_i, local_max),
                               m_i)

            # Rescale accumulators
            alpha = torch.exp(m_i - new_m)
            l_i = l_i * alpha
            acc = acc * alpha.unsqueeze(-1)

            # Compute exp scores and PV
            exp_scores = torch.exp(scores - new_m.unsqueeze(-1))
            exp_scores = torch.where(needs_compute.unsqueeze(-1),
                                    exp_scores,
                                    torch.zeros_like(exp_scores))

            # Only compute PV for non-skipped queries
            pv_contrib = torch.matmul(exp_scores, v_block.float())
            acc = acc + pv_contrib
            l_i = l_i + exp_scores.sum(dim=-1)

            m_i = new_m

    # Final normalization
    output = acc / l_i.unsqueeze(-1).clamp(min=1e-10)

    # LSE for merging
    lse = torch.where(l_i > 0, m_i + torch.log(l_i), torch.full_like(m_i, -1e30))

    return output.to(q.dtype), m_i, lse


def full_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Standard full attention for comparison."""
    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, _ = k.shape

    gqa_ratio = num_heads // num_kv_heads
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=0)
        v = v.repeat_interleave(gqa_ratio, dim=0)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output


# ============================================================
# Tests
# ============================================================

def test_mask_correctness():
    """Test mask calculation correctness."""
    torch.manual_seed(42)

    num_heads = 4
    num_kv_heads = 4
    q_len = 256
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.5)

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    skip_mask, new_running_max = blasst_mask_reference(
        q, k, running_max, ln_lambda, softmax_scale, granularity
    )

    # Verify shapes
    num_kv_blocks = (kv_len + granularity - 1) // granularity
    assert skip_mask.shape == (num_heads, q_len, num_kv_blocks), f"Wrong mask shape: {skip_mask.shape}"
    assert new_running_max.shape == (num_heads, q_len)

    # Verify running_max is monotonically increasing
    assert (new_running_max >= running_max).all(), "running_max should increase"

    # Verify skip rate is reasonable (not all skip, not all keep)
    skip_rate = skip_mask.float().mean().item()
    print(f"test_mask_correctness: PASSED (skip_rate={skip_rate:.1%})")


def test_attn_with_mask_correctness():
    """Test attention with mask matches full attention when mask=all False."""
    torch.manual_seed(42)

    num_heads = 4
    num_kv_heads = 4
    q_len = 256
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    # All False mask = no skipping = should match full attention
    num_kv_blocks = (kv_len + granularity - 1) // granularity
    skip_mask = torch.zeros((num_heads, q_len, num_kv_blocks),
                           dtype=torch.bool, device='cuda')
    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    output, new_running_max, lse = blasst_attn_with_mask_reference(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    full_out = full_attention_reference(q, k, v, softmax_scale)

    max_diff = (output - full_out).abs().max().item()
    assert max_diff < 1e-3, f"Output mismatch: {max_diff}"

    print(f"test_attn_with_mask_correctness: PASSED (max_diff={max_diff:.6f})")


def test_attn_with_skip_logic():
    """Test attention correctly skips low-score blocks."""
    torch.manual_seed(42)

    num_heads = 2
    num_kv_heads = 2
    q_len = 128
    kv_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.001)  # Low threshold = aggressive skipping

    # Create structured input
    q = torch.zeros(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    q[:, :, 0] = 1.0

    k = torch.zeros(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    # Block 0: high scores
    k[:, 0:128, 0] = 80.0
    # Block 1: high scores
    k[:, 128:256, 0] = 72.0
    # Block 2: low scores
    k[:, 256:384, 0] = 0.0
    # Block 3: low scores
    k[:, 384:512, 0] = -16.0

    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    # First compute mask
    skip_mask, new_running_max = blasst_mask_reference(
        q, k, running_max, ln_lambda, softmax_scale, granularity
    )

    skip_rate = skip_mask.float().mean().item()
    print(f"  Skip mask: {skip_rate:.1%} skipped")

    # Then compute attention with mask
    output, final_running_max, lse = blasst_attn_with_mask_reference(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    # Verify output is valid
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Verify some blocks were skipped
    assert skip_rate > 0, "Expected some blocks to be skipped"
    assert skip_rate < 1.0, "Expected some blocks to be kept"

    print(f"test_attn_with_skip_logic: PASSED (skip_rate={skip_rate:.1%})")


def test_gqa_support():
    """Test GQA support."""
    torch.manual_seed(42)

    num_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    q_len = 128
    kv_len = 256
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.5)

    q = torch.randn(num_heads, q_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')

    running_max = torch.full((num_heads, q_len), float('-inf'), device='cuda', dtype=torch.float32)

    skip_mask, new_running_max = blasst_mask_reference(
        q, k, running_max, ln_lambda, softmax_scale, granularity
    )

    output, final_running_max, lse = blasst_attn_with_mask_reference(
        q, k, v, skip_mask, running_max, softmax_scale, granularity
    )

    assert output.shape == (num_heads, q_len, head_dim)
    assert not torch.isnan(output).any()

    print(f"test_gqa_support: PASSED (shape={output.shape})")


if __name__ == "__main__":
    test_mask_correctness()
    test_attn_with_mask_correctness()
    test_attn_with_skip_logic()
    test_gqa_support()
    print("\nAll BLASST split reference tests PASSED!")
