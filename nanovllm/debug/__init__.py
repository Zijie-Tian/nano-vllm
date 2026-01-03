"""
Breakpoint debugging tools for aligning nanovllm with reference implementations.

This module provides a generator-based breakpoint aligner that enables step-by-step
comparison between nanovllm and torch reference model outputs.

Example:
    >>> from nanovllm.debug import BreakpointAligner, TorchSteppable, NanovllmSteppable
    >>> from tests.modeling_qwen3 import Qwen3ForCausalLM
    >>>
    >>> # Load models
    >>> torch_model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.float16)
    >>> nanovllm_model = ...  # Your nanovllm model
    >>>
    >>> # Create adapters
    >>> ref = TorchSteppable(torch_model)
    >>> test = NanovllmSteppable(nanovllm_model)
    >>>
    >>> # Run alignment
    >>> aligner = BreakpointAligner(ref, test)
    >>> result = aligner.align(input_ids)
    >>> print(result)
"""

from .breakpoints import BreakpointType, Breakpoint
from .comparator import TensorComparator, ComparisonResult
from .aligner import BreakpointAligner, AlignmentResult
from .adapters import SteppableModel, TorchSteppable, NanovllmSteppable
from .utils import setup_prefill_context, setup_decode_context, cleanup_context

__all__ = [
    # Core classes
    "BreakpointAligner",
    "AlignmentResult",
    # Breakpoints
    "BreakpointType",
    "Breakpoint",
    # Comparator
    "TensorComparator",
    "ComparisonResult",
    # Adapters
    "SteppableModel",
    "TorchSteppable",
    "NanovllmSteppable",
    # Utils
    "setup_prefill_context",
    "setup_decode_context",
    "cleanup_context",
]
