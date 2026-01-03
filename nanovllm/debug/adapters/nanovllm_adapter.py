"""Nanovllm model adapter for breakpoint alignment."""

from typing import Generator, Set, Optional, Dict, Any, List
import torch

from nanovllm.utils.context import set_context, reset_context
from ..breakpoints import Breakpoint, BreakpointType
from .base import SteppableModel


class NanovllmSteppable(SteppableModel):
    """
    Steppable adapter for nanovllm Qwen3 implementation.

    Uses PyTorch hooks to capture intermediate values during forward pass,
    then yields them as breakpoints after execution completes.

    Key challenges handled:
    1. Shape difference: nanovllm uses [num_tokens, hidden] vs [batch, seq, hidden]
    2. Context-based attention: must call set_context() before forward
    3. Fused operations: decoder layer returns (hidden_states, residual) tuple
    """

    def __init__(
        self,
        model,
        enabled_breakpoints: Optional[Set[BreakpointType]] = None,
    ):
        """
        Args:
            model: Qwen3ForCausalLM from nanovllm
            enabled_breakpoints: Set of breakpoint types to yield at
        """
        super().__init__(enabled_breakpoints)
        self.model = model
        self.model.eval()
        self._hooks: List[Any] = []
        self._captured: Dict[str, torch.Tensor] = {}

    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def _register_hooks(self):
        """Register forward hooks on all relevant modules."""
        self._hooks = []
        self._captured = {}

        # Hook for embedding output
        def embed_hook(module, input, output):
            self._captured["embed"] = output.detach().clone()

        self._hooks.append(
            self.model.model.embed_tokens.register_forward_hook(embed_hook)
        )

        # Hooks for each decoder layer
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            def make_layer_hook(idx):
                def hook(module, input, output):
                    # Decoder layer returns (hidden_states, residual)
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self._captured[f"layer_{idx}"] = hidden_states.detach().clone()
                return hook

            self._hooks.append(
                layer.register_forward_hook(make_layer_hook(layer_idx))
            )

        # Hook for final norm
        def final_norm_hook(module, input, output):
            # Final norm returns (hidden_states, _) for fused add
            hidden_states = output[0] if isinstance(output, tuple) else output
            self._captured["final_norm"] = hidden_states.detach().clone()

        self._hooks.append(
            self.model.model.norm.register_forward_hook(final_norm_hook)
        )

        # Hook for lm_head
        def lm_head_hook(module, input, output):
            self._captured["lm_head"] = output.detach().clone()

        self._hooks.append(
            self.model.lm_head.register_forward_hook(lm_head_hook)
        )

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _setup_context(self, seq_len: int, device: torch.device, is_prefill: bool):
        """
        Set up nanovllm context for forward pass.

        For alignment testing, we use simple context without real KV cache.
        """
        if is_prefill:
            # Prefill: process all tokens at once
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            # Use -1 for slot_mapping to skip KV cache writes
            slot_mapping = torch.full((seq_len,), -1, dtype=torch.int32, device=device)

            set_context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                slot_mapping=slot_mapping,
                is_chunked_prefill=False,
            )
        else:
            # Decode: single token generation
            # For decode, we need context_lens and block_tables
            # For alignment testing without real KV cache, we use minimal setup
            context_lens = torch.tensor([seq_len - 1], dtype=torch.int32, device=device)
            # Single token slot
            slot_mapping = torch.tensor([-1], dtype=torch.int32, device=device)
            # Empty block tables (no KV cache)
            block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)

            set_context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize nanovllm tensor shape to [batch, seq_len, ...].

        nanovllm uses [num_tokens, ...] format without batch dimension.
        We add batch dimension for comparison with torch model.
        """
        if tensor.dim() == 2:  # [num_tokens, hidden_size]
            return tensor.unsqueeze(0)
        return tensor

    def step(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> Generator[Breakpoint, None, torch.Tensor]:
        """
        Execute nanovllm forward pass with hooks to capture breakpoints.

        Unlike the torch adapter which manually steps through each component,
        we run the full forward pass and collect captured values afterward.
        """
        # Ensure 1D for nanovllm (it expects [num_tokens])
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)

        seq_len = input_ids.numel()
        device = input_ids.device

        # Generate position IDs if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device)
        elif positions.dim() == 2:
            positions = positions.squeeze(0)

        # Register hooks
        self._register_hooks()

        try:
            # Setup context for attention
            self._setup_context(seq_len, device, is_prefill)

            # Run forward pass (hooks capture everything)
            with torch.no_grad():
                hidden_states = self.model(input_ids, positions)
                logits = self.model.compute_logits(hidden_states)

            reset_context()

            # Yield breakpoints in order from captured data

            # EMBEDDING
            if self.is_enabled(BreakpointType.EMBEDDING) and "embed" in self._captured:
                yield Breakpoint(
                    bp_type=BreakpointType.EMBEDDING,
                    layer_idx=None,
                    tensor=self._normalize_tensor(self._captured["embed"]),
                    name="Embedding",
                )

            # LAYER_OUTPUT for each layer
            if self.is_enabled(BreakpointType.LAYER_OUTPUT):
                for layer_idx in range(self.num_layers):
                    key = f"layer_{layer_idx}"
                    if key in self._captured:
                        yield Breakpoint(
                            bp_type=BreakpointType.LAYER_OUTPUT,
                            layer_idx=layer_idx,
                            tensor=self._normalize_tensor(self._captured[key]),
                            name=f"Layer {layer_idx}",
                        )

            # FINAL_NORM
            if self.is_enabled(BreakpointType.FINAL_NORM) and "final_norm" in self._captured:
                yield Breakpoint(
                    bp_type=BreakpointType.FINAL_NORM,
                    layer_idx=None,
                    tensor=self._normalize_tensor(self._captured["final_norm"]),
                    name="Final Norm",
                )

            # LM_HEAD
            if self.is_enabled(BreakpointType.LM_HEAD) and "lm_head" in self._captured:
                yield Breakpoint(
                    bp_type=BreakpointType.LM_HEAD,
                    layer_idx=None,
                    tensor=self._normalize_tensor(self._captured["lm_head"]),
                    name="LM Head",
                )

            return logits

        finally:
            self._remove_hooks()
            self._captured = {}
