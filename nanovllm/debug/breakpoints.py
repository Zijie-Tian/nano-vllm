"""Breakpoint types and data structures for alignment debugging."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import torch


class BreakpointType(Enum):
    """Types of breakpoints in the model forward pass."""
    EMBEDDING = auto()      # After embed_tokens
    LAYER_OUTPUT = auto()   # After each decoder layer
    FINAL_NORM = auto()     # After final RMSNorm
    LM_HEAD = auto()        # After lm_head (logits)


@dataclass
class Breakpoint:
    """A captured breakpoint with tensor data."""
    bp_type: BreakpointType
    layer_idx: Optional[int]  # None for EMBEDDING, FINAL_NORM, LM_HEAD
    tensor: torch.Tensor
    name: str

    def normalize_shape(self) -> torch.Tensor:
        """
        Normalize tensor shape for comparison.

        nanovllm uses [num_tokens, hidden_size] while torch uses
        [batch, seq_len, hidden_size]. This adds a batch dimension
        to 2D tensors for comparison.
        """
        if self.tensor.dim() == 2:
            return self.tensor.unsqueeze(0)
        return self.tensor

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) for d in self.tensor.shape)
        return f"Breakpoint({self.name}, shape={shape_str}, dtype={self.tensor.dtype})"
