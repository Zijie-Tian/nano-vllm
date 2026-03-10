"""Breakpoint aligner for comparing model outputs."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

from .breakpoints import Breakpoint
from .comparator import TensorComparator, ComparisonResult
from .adapters.base import SteppableModel


@dataclass
class AlignmentResult:
    """Result of an alignment test."""

    passed: bool
    all_comparisons: List[Tuple[Breakpoint, Breakpoint, ComparisonResult]] = field(
        default_factory=list
    )
    failed_at: Optional[Breakpoint] = None
    message: str = ""

    def __repr__(self) -> str:
        passed_count = sum(1 for _, _, c in self.all_comparisons if c.passed)
        total = len(self.all_comparisons)
        status = "PASSED" if self.passed else "FAILED"
        return f"AlignmentResult({status}, {passed_count}/{total} breakpoints passed)"


class BreakpointAligner:
    """
    Orchestrates alternating execution of reference and test models,
    comparing outputs at each breakpoint.

    Example:
        >>> from nanovllm.debug import BreakpointAligner, TorchSteppable, NanovllmSteppable
        >>> ref = TorchSteppable(torch_model)
        >>> test = NanovllmSteppable(nanovllm_model)
        >>> aligner = BreakpointAligner(ref, test)
        >>> result = aligner.align(input_ids)
        >>> print(result)
    """

    def __init__(
        self,
        ref_model: SteppableModel,
        test_model: SteppableModel,
        comparator: Optional[TensorComparator] = None,
        stop_on_error: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            ref_model: Reference (torch) steppable model
            test_model: Test (nanovllm) steppable model
            comparator: Tensor comparator instance (uses default if None)
            stop_on_error: If True, stop at first mismatch
            verbose: If True, print comparison results
        """
        self.ref_model = ref_model
        self.test_model = test_model
        self.comparator = comparator or TensorComparator()
        self.stop_on_error = stop_on_error
        self.verbose = verbose

    def align(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> AlignmentResult:
        """
        Run both models with same input, comparing at each breakpoint.

        Args:
            input_ids: Input token IDs
            positions: Position IDs (optional)
            is_prefill: True for prefill phase, False for decode

        Returns:
            AlignmentResult with pass/fail status and details
        """
        all_comparisons: List[Tuple[Breakpoint, Breakpoint, ComparisonResult]] = []

        if self.verbose:
            phase = "prefill" if is_prefill else "decode"
            print(f"\n{'=' * 60}")
            print(f"Alignment Test ({phase})")
            print(f"{'=' * 60}")

        # Start both generators
        ref_gen = self.ref_model.step(input_ids, positions, is_prefill)
        test_gen = self.test_model.step(input_ids, positions, is_prefill)

        try:
            while True:
                # Get next breakpoint from reference
                try:
                    ref_bp = next(ref_gen)
                except StopIteration:
                    break

                # Get corresponding breakpoint from test
                try:
                    test_bp = next(test_gen)
                except StopIteration:
                    if self.verbose:
                        print(f"Test model ended early at {ref_bp.name}")
                    return AlignmentResult(
                        passed=False,
                        all_comparisons=all_comparisons,
                        failed_at=ref_bp,
                        message=f"Test model ended early at {ref_bp.name}",
                    )

                # Verify breakpoints match
                if ref_bp.bp_type != test_bp.bp_type:
                    msg = f"Breakpoint type mismatch: {ref_bp.bp_type} vs {test_bp.bp_type}"
                    if self.verbose:
                        print(msg)
                    return AlignmentResult(
                        passed=False,
                        all_comparisons=all_comparisons,
                        failed_at=ref_bp,
                        message=msg,
                    )

                if ref_bp.layer_idx != test_bp.layer_idx:
                    msg = f"Layer index mismatch: {ref_bp.layer_idx} vs {test_bp.layer_idx}"
                    if self.verbose:
                        print(msg)
                    return AlignmentResult(
                        passed=False,
                        all_comparisons=all_comparisons,
                        failed_at=ref_bp,
                        message=msg,
                    )

                # Normalize shapes for comparison
                ref_t = ref_bp.normalize_shape()
                test_t = test_bp.normalize_shape()

                # Handle shape mismatches
                if ref_t.shape != test_t.shape:
                    if self.verbose:
                        print(
                            f"[{ref_bp.name}] Shape mismatch: ref={ref_t.shape} vs test={test_t.shape}"
                        )

                    # Try to reshape if element count matches
                    if ref_t.numel() == test_t.numel():
                        test_t = test_t.view(ref_t.shape)
                    else:
                        msg = f"Shape mismatch at {ref_bp.name}: {ref_t.shape} vs {test_t.shape}"
                        return AlignmentResult(
                            passed=False,
                            all_comparisons=all_comparisons,
                            failed_at=ref_bp,
                            message=msg,
                        )

                # Compare tensors
                result = self.comparator.compare(ref_t, test_t, ref_bp.name)
                all_comparisons.append((ref_bp, test_bp, result))

                if self.verbose:
                    status = "\u2713" if result.passed else "\u2717"
                    print(
                        f"{status} [{ref_bp.name}] cos={result.cosine_similarity:.6f}, max_diff={result.max_abs_diff:.2e}"
                    )

                if not result.passed and self.stop_on_error:
                    if self.verbose:
                        print(f"\nStopped at {ref_bp.name} (stop_on_error=True)")
                        print(result.message)
                    return AlignmentResult(
                        passed=False,
                        all_comparisons=all_comparisons,
                        failed_at=ref_bp,
                        message=f"Alignment failed at {ref_bp.name}",
                    )

            # Check for extra test breakpoints
            try:
                extra_bp = next(test_gen)
                msg = f"Test model has extra breakpoints starting at {extra_bp.name}"
                if self.verbose:
                    print(msg)
                return AlignmentResult(
                    passed=False,
                    all_comparisons=all_comparisons,
                    message=msg,
                )
            except StopIteration:
                pass

        except Exception as e:
            msg = f"Exception during alignment: {str(e)}"
            if self.verbose:
                print(msg)
            raise

        # Summary
        all_passed = all(comp[2].passed for comp in all_comparisons)
        passed_count = sum(1 for _, _, c in all_comparisons if c.passed)
        total = len(all_comparisons)

        if self.verbose:
            print(f"{'=' * 60}")
            status = "PASSED" if all_passed else "FAILED"
            print(f"Result: {status} ({passed_count}/{total} breakpoints)")
            print(f"{'=' * 60}\n")

        return AlignmentResult(
            passed=all_passed,
            all_comparisons=all_comparisons,
            message="All breakpoints aligned"
            if all_passed
            else "Some breakpoints failed",
        )
