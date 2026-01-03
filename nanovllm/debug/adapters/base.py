"""Base class for steppable model adapters."""

from abc import ABC, abstractmethod
from typing import Generator, Set, Optional
import torch

from ..breakpoints import Breakpoint, BreakpointType


class SteppableModel(ABC):
    """
    Abstract base class for models that can yield at breakpoints.

    Subclasses implement the step() method as a generator that yields
    Breakpoint objects at each enabled breakpoint during forward pass.
    """

    def __init__(self, enabled_breakpoints: Optional[Set[BreakpointType]] = None):
        """
        Args:
            enabled_breakpoints: Set of breakpoint types to yield at.
                                If None, yields at all breakpoints.
        """
        self.enabled_breakpoints = enabled_breakpoints

    def is_enabled(self, bp_type: BreakpointType) -> bool:
        """Check if a breakpoint type is enabled."""
        if self.enabled_breakpoints is None:
            return True
        return bp_type in self.enabled_breakpoints

    @abstractmethod
    def step(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> Generator[Breakpoint, None, torch.Tensor]:
        """
        Generator that yields Breakpoint objects at enabled breakpoints.

        Args:
            input_ids: Input token IDs
            positions: Position IDs (optional, auto-generated if None)
            is_prefill: True for prefill phase, False for decode

        Yields:
            Breakpoint objects at each enabled checkpoint

        Returns:
            Final output tensor (logits)
        """
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of decoder layers."""
        pass
