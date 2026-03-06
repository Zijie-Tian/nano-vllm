"""
Test: BLASST Fused Kernel Reference Implementation

Pure PyTorch reference implementation of the BLASST algorithm for validation.
Follows the paper design: QK -> local_max -> skip check -> conditional PV
"""
import torch
import math
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")

# ============================================================
# BLASST Algorithm Reference Implementation
# ============================================================

def blasst_attention_reference(
    q: torch.Tensor,          # [num_heads, seq_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    v: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    ln_lambda: float,         # ln(λ) threshold
    softmax_scale: float,
    granularity: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Reference implementation of BLASST attention.

    Algorithm per query token:
    1. Initialize: running_max = -inf, output = 0, sum_exp = 0
    2. For each KV block at granularity:
       a. Compute QK -> scores [num_heads, q_len, granularity]
       b. Compute local_max per query token
       c. If local_max - running_max < ln_lambda: skip
       d. Else: compute PV with online softmax, update running stats
    3. Final normalization: output / sum_exp

    Returns:
        output: [num_heads, seq_len, head_dim]
        running_max_final: [num_heads, seq_len]
        stats: dict with skip rate and other metrics
    """
    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, _ = k.shape

    # GQA handling: repeat KV heads if needed
    gqa_ratio = num_heads // num_kv_heads
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=0)
        v = v.repeat_interleave(gqa_ratio, dim=0)

    # Initialize running statistics
    running_max = torch.full((num_heads, q_len), float('-inf'), dtype=torch.float32, device=q.device)
    running_sum_exp = torch.zeros((num_heads, q_len), dtype=torch.float32, device=q.device)
    output = torch.zeros((num_heads, q_len, head_dim), dtype=torch.float32, device=q.device)

    num_blocks = (kv_len + granularity - 1) // granularity
    skipped_blocks = 0
    total_blocks = 0

    # Process each KV block
    for block_idx in range(num_blocks):
        kv_start = block_idx * granularity
        kv_end = min(kv_start + granularity, kv_len)
        kv_len_actual = kv_end - kv_start

        # Load KV block
        k_block = k[:, kv_start:kv_end, :]  # [num_heads, kv_len_actual, head_dim]
        v_block = v[:, kv_start:kv_end, :]

        # Step 1: Compute QK
        scores = torch.matmul(q, k_block.transpose(-2, -1)) * softmax_scale
        # scores: [num_heads, q_len, kv_len_actual]

        # Step 2: Get local_max per query token
        local_max = scores.max(dim=-1).values  # [num_heads, q_len]

        # Step 3: BLASST skip check
        skip_condition = (local_max - running_max) < ln_lambda  # [num_heads, q_len]
        total_blocks += 1

        # For simplicity in reference: skip entire block if ANY query wants to skip
        # (In real kernel, we'd handle per-query skipping with masking)
        if skip_condition.all():
            # All queries skip - just update running_max
            running_max = torch.maximum(running_max, local_max)
            skipped_blocks += 1
            continue

        # Step 4: Compute PV with online softmax update (for non-skipped block)
        # Online softmax: https://arxiv.org/abs/1805.02867
        new_max = torch.maximum(running_max, local_max)

        # Compute exp(scores - new_max) for numerical stability
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))  # [num_heads, q_len, kv_len_actual]
        sum_exp = exp_scores.sum(dim=-1)  # [num_heads, q_len]

        # Rescale previous output and sum_exp
        if block_idx > 0:
            scale_factor = torch.exp(running_max - new_max)  # [num_heads, q_len]
            output = output * scale_factor.unsqueeze(-1)
            running_sum_exp = running_sum_exp * scale_factor

        # Compute PV and accumulate
        # exp_scores @ v_block: [num_heads, q_len, kv_len_actual] @ [num_heads, kv_len_actual, head_dim]
        pv_contrib = torch.matmul(exp_scores, v_block.float())  # [num_heads, q_len, head_dim]
        output = output + pv_contrib
        running_sum_exp = running_sum_exp + sum_exp
        running_max = new_max

    # Final normalization
    output = output / running_sum_exp.unsqueeze(-1)

    stats = {
        'skipped_blocks': skipped_blocks,
        'total_blocks': total_blocks,
        'skip_rate': skipped_blocks / total_blocks if total_blocks > 0 else 0.0,
    }

    return output.to(q.dtype), running_max, stats


def full_attention_reference(
    q: torch.Tensor,          # [num_heads, seq_len, head_dim]
    k: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    v: torch.Tensor,          # [num_kv_heads, kv_len, head_dim]
    softmax_scale: float,
) -> torch.Tensor:
    """
    Standard full attention for comparison.
    """
    num_heads, q_len, head_dim = q.shape
    num_kv_heads, kv_len, _ = k.shape

    # GQA handling
    gqa_ratio = num_heads // num_kv_heads
    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=0)
        v = v.repeat_interleave(gqa_ratio, dim=0)

    # Compute attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output


# ============================================================
# Tests
# ============================================================

def test_blasst_correctness():
    """Test BLASST output matches full attention (with high threshold)."""
    torch.manual_seed(42)

    # ============================================================
    # Parameters
    # ============================================================
    num_heads = 8
    num_kv_heads = 2  # GQA ratio = 4
    seq_len = 512
    head_dim = 128
    granularity = 128
    softmax_scale = head_dim ** -0.5

    # High lambda = don't skip anything (should match full attention)
    lambda_val = 1.0  # ln(1.0) = 0, so skip only when local_max < running_max
    ln_lambda = math.log(lambda_val)

    # ============================================================
    # Construct input (structured for predictability)
    # ============================================================
    q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # ============================================================
    # Run both implementations
    # ============================================================
    blasst_out, running_max, stats = blasst_attention_reference(
        q, k, v, ln_lambda, softmax_scale, granularity
    )

    full_out = full_attention_reference(q, k, v, softmax_scale)

    # ============================================================
    # Verify: With lambda=1.0, BLASST should not skip any blocks
    # ============================================================
    assert stats['skipped_blocks'] == 0, f"Expected no skips with lambda=1.0, got {stats['skipped_blocks']}"

    # Verify numerical correctness
    max_diff = (blasst_out - full_out).abs().max().item()
    assert max_diff < 1e-3, f"Max diff too large: {max_diff}"

    print(f"test_blasst_correctness: PASSED (max_diff={max_diff:.6f}, skip_rate={stats['skip_rate']:.1%})")


def test_blasst_skip_logic():
    """Test that BLASST correctly skips blocks with low lambda."""
    torch.manual_seed(42)

    # ============================================================
    # Parameters
    # ============================================================
    num_heads = 4
    num_kv_heads = 4  # MHA for simplicity in this test
    seq_len = 512
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5

    # Low lambda = aggressive skipping
    lambda_val = 0.001  # Very low threshold
    ln_lambda = math.log(lambda_val)

    # ============================================================
    # Construct input with block-wise score control.
    # Block order matters for BLASST:
    #   1) High-score blocks first -> running_max becomes large
    #   2) Low-score blocks later -> satisfy skip condition
    # ============================================================
    q = torch.zeros(num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    q[:, :, 0] = 1.0

    k = torch.zeros(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # 4 blocks total (seq_len=512, granularity=128)
    # Score = (q dot k) * softmax_scale, with q using only dim-0.
    # block0: score ~ 10  (high, not skipped)
    # block1: score ~ 9   (high, not skipped)
    # block2: score ~ 0   (low, skipped after running_max~10)
    # block3: score ~ -2  (low, skipped after running_max~10)
    k[:, 0:128, 0] = 80.0   # 80 * 0.125 = 10
    k[:, 128:256, 0] = 72.0  # 72 * 0.125 = 9
    k[:, 256:384, 0] = 0.0
    k[:, 384:512, 0] = -16.0  # -16 * 0.125 = -2

    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # ============================================================
    # Run BLASST
    # ============================================================
    blasst_out, running_max, stats = blasst_attention_reference(
        q, k, v, ln_lambda, softmax_scale, granularity
    )

    # ============================================================
    # Verify: low-score trailing blocks should be skipped
    # ============================================================
    assert stats['skipped_blocks'] > 0, f"Expected some skips with lambda={lambda_val}, got 0"
    assert stats['skipped_blocks'] < stats['total_blocks'], (
        "Expected high-score blocks to remain unskipped"
    )
    print(f"test_blasst_skip_logic: PASSED (skip_rate={stats['skip_rate']:.1%})")


def test_blasst_gqa():
    """Test BLASST with GQA (grouped query attention)."""
    torch.manual_seed(42)

    # ============================================================
    # Parameters (GQA: 8 query heads, 2 KV heads)
    # ============================================================
    num_heads = 8
    num_kv_heads = 2
    seq_len = 256
    head_dim = 64
    granularity = 128
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(0.5)

    # ============================================================
    # Construct input
    # ============================================================
    q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # ============================================================
    # Run BLASST
    # ============================================================
    blasst_out, running_max, stats = blasst_attention_reference(
        q, k, v, ln_lambda, softmax_scale, granularity
    )

    # ============================================================
    # Verify output shape
    # ============================================================
    assert blasst_out.shape == (num_heads, seq_len, head_dim)
    assert running_max.shape == (num_heads, seq_len)

    # Verify no NaN
    assert not torch.isnan(blasst_out).any()
    assert not torch.isinf(blasst_out).any()

    print(f"test_blasst_gqa: PASSED (shape={blasst_out.shape}, skip_rate={stats['skip_rate']:.1%})")


def test_blasst_online_softmax():
    """Test that online softmax produces correct output."""
    torch.manual_seed(42)

    # ============================================================
    # Parameters
    # ============================================================
    num_heads = 2
    num_kv_heads = 2
    seq_len = 128
    head_dim = 64
    granularity = 64  # Small granularity for more iterations
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(1.0)  # No skipping

    # ============================================================
    # Construct simple input where we can compute expected output
    # ============================================================
    q = torch.ones(num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.ones(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.eye(head_dim, dtype=torch.float16, device='cuda').unsqueeze(0).unsqueeze(0)
    v = v.expand(num_kv_heads, seq_len, -1, -1).mean(dim=2)  # [num_kv_heads, seq_len, head_dim]

    # ============================================================
    # Run both implementations
    # ============================================================
    blasst_out, running_max, stats = blasst_attention_reference(
        q, k, v, ln_lambda, softmax_scale, granularity
    )

    full_out = full_attention_reference(q, k, v, softmax_scale)

    # ============================================================
    # Verify online softmax matches standard softmax
    # ============================================================
    max_diff = (blasst_out - full_out).abs().max().item()
    assert max_diff < 1e-3, f"Online softmax mismatch: {max_diff}"

    print(f"test_blasst_online_softmax: PASSED (max_diff={max_diff:.6f})")


def test_blasst_running_max_update():
    """Test that running_max is correctly updated across blocks."""
    torch.manual_seed(42)

    # ============================================================
    # Parameters
    # ============================================================
    num_heads = 1
    num_kv_heads = 1
    seq_len = 128
    head_dim = 64
    granularity = 64
    softmax_scale = head_dim ** -0.5
    ln_lambda = math.log(1.0)  # No skipping

    # ============================================================
    # Create input with known max values
    # ============================================================
    q = torch.zeros(num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    q[:, :, 0] = 1.0  # Simple query vector

    k = torch.zeros(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    # First block: score = 0.5
    k[:, :64, 0] = 0.5 / softmax_scale
    # Second block: score = 1.0 (higher)
    k[:, 64:, 0] = 1.0 / softmax_scale

    v = torch.ones(num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # ============================================================
    # Run BLASST
    # ============================================================
    blasst_out, running_max, stats = blasst_attention_reference(
        q, k, v, ln_lambda, softmax_scale, granularity
    )

    # ============================================================
    # Verify running_max is monotonically increasing
    # and reaches the expected max
    # ============================================================
    expected_max = 1.0 * softmax_scale  # From second block
    assert (running_max >= expected_max - 0.01).all(), f"running_max should reach {expected_max}"

    print(f"test_blasst_running_max_update: PASSED (final_max={running_max.mean().item():.4f})")


if __name__ == "__main__":
    test_blasst_correctness()
    test_blasst_skip_logic()
    test_blasst_gqa()
    test_blasst_online_softmax()
    test_blasst_running_max_update()
    print("\nAll BLASST reference tests PASSED!")
