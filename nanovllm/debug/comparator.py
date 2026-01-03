"""Tensor comparison utilities for alignment debugging."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ComparisonResult:
    """Result of comparing two tensors."""
    passed: bool
    cosine_similarity: float
    max_abs_diff: float
    mean_abs_diff: float
    message: str

    def __repr__(self) -> str:
        status = "\u2713" if self.passed else "\u2717"
        return f"{status} cos={self.cosine_similarity:.6f}, max_diff={self.max_abs_diff:.2e}"


class TensorComparator:
    """Compares tensors using cosine similarity and absolute differences."""

    def __init__(
        self,
        cosine_threshold: float = 0.999,
        max_diff_threshold: float = 0.1,
        mean_diff_threshold: float = 0.01,
    ):
        """
        Args:
            cosine_threshold: Minimum cosine similarity to pass (0-1)
            max_diff_threshold: Maximum allowed absolute difference
            mean_diff_threshold: Maximum allowed mean absolute difference
        """
        self.cosine_threshold = cosine_threshold
        self.max_diff_threshold = max_diff_threshold
        self.mean_diff_threshold = mean_diff_threshold

    def compare(
        self,
        ref: torch.Tensor,
        test: torch.Tensor,
        name: str = "",
    ) -> ComparisonResult:
        """
        Compare two tensors and return detailed result.

        Args:
            ref: Reference tensor
            test: Test tensor
            name: Name for the comparison (used in message)

        Returns:
            ComparisonResult with pass/fail status and metrics
        """
        # Convert to float32 for comparison
        ref_f = ref.float().flatten()
        test_f = test.float().flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            ref_f.unsqueeze(0),
            test_f.unsqueeze(0)
        ).item()

        # Absolute differences
        diff = (ref.float() - test.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check thresholds
        passed = (
            cos_sim >= self.cosine_threshold and
            max_diff <= self.max_diff_threshold and
            mean_diff <= self.mean_diff_threshold
        )

        status = "PASS" if passed else "FAIL"
        message = (
            f"[{name}] {status}\n"
            f"  Cosine Similarity: {cos_sim:.6f} (threshold: {self.cosine_threshold})\n"
            f"  Max Abs Diff: {max_diff:.6f} (threshold: {self.max_diff_threshold})\n"
            f"  Mean Abs Diff: {mean_diff:.6f} (threshold: {self.mean_diff_threshold})"
        )

        return ComparisonResult(
            passed=passed,
            cosine_similarity=cos_sim,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            message=message,
        )
