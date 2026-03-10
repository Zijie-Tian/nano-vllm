"""Torch reference model adapter for breakpoint alignment."""

from typing import Generator, Set, Optional
import torch

from ..breakpoints import Breakpoint, BreakpointType
from .base import SteppableModel


class TorchSteppable(SteppableModel):
    """
    Steppable adapter for the torch reference Qwen3 implementation.

    Wraps tests/modeling_qwen3.py Qwen3ForCausalLM model.
    """

    def __init__(
        self,
        model,
        enabled_breakpoints: Optional[Set[BreakpointType]] = None,
    ):
        """
        Args:
            model: Qwen3ForCausalLM from tests/modeling_qwen3.py
            enabled_breakpoints: Set of breakpoint types to yield at
        """
        super().__init__(enabled_breakpoints)
        self.model = model
        self.model.eval()

    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def step(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> Generator[Breakpoint, None, torch.Tensor]:
        """
        Generator that manually steps through the torch model.

        The torch model uses [batch, seq_len, hidden_size] shapes.
        """
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate position IDs if not provided
        if positions is None:
            positions = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )
        elif positions.dim() == 1:
            positions = positions.unsqueeze(0)

        with torch.no_grad():
            # === EMBEDDING ===
            hidden_states = self.model.model.embed_tokens(input_ids)

            if self.is_enabled(BreakpointType.EMBEDDING):
                yield Breakpoint(
                    bp_type=BreakpointType.EMBEDDING,
                    layer_idx=None,
                    tensor=hidden_states.detach().clone(),
                    name="Embedding",
                )

            # Create causal attention mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # === DECODER LAYERS ===
            for layer_idx, layer in enumerate(self.model.model.layers):
                hidden_states, _, _ = layer(
                    hidden_states=hidden_states,
                    position_ids=positions,
                    attention_mask=attention_mask,
                    past_key_value=None,
                    use_cache=False,
                    output_qkv=False,
                )

                if self.is_enabled(BreakpointType.LAYER_OUTPUT):
                    yield Breakpoint(
                        bp_type=BreakpointType.LAYER_OUTPUT,
                        layer_idx=layer_idx,
                        tensor=hidden_states.detach().clone(),
                        name=f"Layer {layer_idx}",
                    )

            # === FINAL NORM ===
            hidden_states = self.model.model.norm(hidden_states)

            if self.is_enabled(BreakpointType.FINAL_NORM):
                yield Breakpoint(
                    bp_type=BreakpointType.FINAL_NORM,
                    layer_idx=None,
                    tensor=hidden_states.detach().clone(),
                    name="Final Norm",
                )

            # === LM HEAD ===
            logits = self.model.lm_head(hidden_states)

            if self.is_enabled(BreakpointType.LM_HEAD):
                yield Breakpoint(
                    bp_type=BreakpointType.LM_HEAD,
                    layer_idx=None,
                    tensor=logits.detach().clone(),
                    name="LM Head",
                )

            return logits
