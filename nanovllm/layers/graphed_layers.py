"""
CUDA Graph wrapped layers for offload optimization.

This module provides Graph-wrapped versions of non-attention layers
to reduce kernel launch overhead in CPU offload path.

Phase 5 Design:
- Supports both Prefill (seq_len=chunk_size) and Decode (seq_len=1)
- Extended coverage: embed, input_norm, qkv_proj, rotary, o_proj, post_norm, mlp, final_norm
- Only attention core (attn.forward) remains in eager mode

Graph Structure (N layers):
- EmbedGraph: embed_tokens
- FirstGraph: input_norm → qkv_proj → rotary
- InterGraph[i]: o_proj → post_norm → mlp → input_norm → qkv_proj → rotary (N-1 graphs)
- LastGraph: o_proj → post_norm → mlp → final_norm
Total: N+2 graphs
"""

import torch
from torch import nn
from typing import Optional, Tuple


class EmbedGraph(nn.Module):
    """
    Graph wrapper for embedding layer.

    Input: input_ids [seq_len]
    Output: hidden_states [seq_len, hidden_size]
    """

    def __init__(
        self,
        embed_tokens: nn.Module,
        seq_len: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dtype = dtype

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.ids_in: Optional[torch.Tensor] = None
        self.h_out: Optional[torch.Tensor] = None

    def _compute(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def capture_graph(self, graph_pool=None):
        """Capture CUDA Graph."""
        # Allocate placeholders outside inference_mode
        self.ids_in = torch.zeros(self.seq_len, dtype=torch.long, device="cuda")
        self.h_out = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                h = self._compute(self.ids_in)
            self.h_out.copy_(h)
            torch.cuda.synchronize()

            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=graph_pool):
                h = self._compute(self.ids_in)
                self.h_out.copy_(h)

        return self.graph.pool() if graph_pool is None else graph_pool

    def forward(self, input_ids: torch.Tensor, use_graph: bool = False) -> torch.Tensor:
        if use_graph and self.graph is not None and input_ids.shape[0] == self.seq_len:
            self.ids_in.copy_(input_ids)
            self.graph.replay()
            return self.h_out.clone()
        else:
            return self._compute(input_ids)


class FirstGraph(nn.Module):
    """
    Graph wrapper for first layer pre-attention:
    input_norm → qkv_proj → split → reshape → rotary

    Input: hidden_states [seq_len, hidden_size], positions [seq_len]
    Output: q [seq_len, num_heads, head_dim], k [seq_len, num_kv_heads, head_dim],
            v [seq_len, num_kv_heads, head_dim], residual [seq_len, hidden_size]
    """

    def __init__(
        self,
        input_norm: nn.Module,
        qkv_proj: nn.Module,
        rotary_emb: nn.Module,
        # Shape parameters
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.input_norm = input_norm
        self.qkv_proj = qkv_proj
        self.rotary_emb = rotary_emb

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Split sizes
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _compute(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        First layer computation:
        1. input_layernorm (residual = hidden_states for first layer)
        2. QKV projection
        3. Split and reshape
        4. Rotary embedding
        """
        # For first layer, residual = hidden_states (before norm)
        residual = hidden_states.clone()
        hidden_states = self.input_norm(hidden_states)

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Rotary embedding
        q, k = self.rotary_emb(positions, q, k)

        return q, k, v, residual

    def capture_graph(self, graph_pool=None):
        """Capture CUDA Graph."""
        # Allocate placeholders
        self.h_in = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")
        self.pos_in = torch.zeros(self.seq_len, dtype=torch.long, device="cuda")

        self.q_out = torch.zeros(self.seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.k_out = torch.zeros(self.seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.v_out = torch.zeros(self.seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.r_out = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                q, k, v, r = self._compute(self.h_in, self.pos_in)
            self.q_out.copy_(q)
            self.k_out.copy_(k)
            self.v_out.copy_(v)
            self.r_out.copy_(r)
            torch.cuda.synchronize()

            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=graph_pool):
                q, k, v, r = self._compute(self.h_in, self.pos_in)
                self.q_out.copy_(q)
                self.k_out.copy_(k)
                self.v_out.copy_(v)
                self.r_out.copy_(r)

        return self.graph.pool() if graph_pool is None else graph_pool

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        use_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_graph and self.graph is not None and hidden_states.shape[0] == self.seq_len:
            self.h_in.copy_(hidden_states)
            self.pos_in.copy_(positions)
            self.graph.replay()
            return self.q_out.clone(), self.k_out.clone(), self.v_out.clone(), self.r_out.clone()
        else:
            return self._compute(hidden_states, positions)


class InterGraph(nn.Module):
    """
    Graph wrapper for inter-layer computation:
    o_proj → post_norm → mlp → input_norm → qkv_proj → rotary

    Merges current layer's post-attention with next layer's pre-attention.

    Input: attn_output [seq_len, num_heads, head_dim], residual [seq_len, hidden_size], positions [seq_len]
    Output: q [seq_len, num_heads, head_dim], k [seq_len, num_kv_heads, head_dim],
            v [seq_len, num_kv_heads, head_dim], residual [seq_len, hidden_size]
    """

    def __init__(
        self,
        # Current layer components
        o_proj: nn.Module,
        post_norm: nn.Module,
        mlp: nn.Module,
        # Next layer components
        next_input_norm: nn.Module,
        next_qkv_proj: nn.Module,
        next_rotary_emb: nn.Module,
        # Shape parameters
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # Current layer
        self.o_proj = o_proj
        self.post_norm = post_norm
        self.mlp = mlp

        # Next layer
        self.next_input_norm = next_input_norm
        self.next_qkv_proj = next_qkv_proj
        self.next_rotary_emb = next_rotary_emb

        # Shape params
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Split sizes
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _compute(
        self,
        attn_output: torch.Tensor,  # [seq_len, num_heads, head_dim]
        residual: torch.Tensor,      # [seq_len, hidden_size]
        positions: torch.Tensor,     # [seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inter-layer computation:
        1. O projection (flatten first)
        2. Post-attention layernorm + residual
        3. MLP
        4. Next layer's input layernorm + residual
        5. QKV projection
        6. Split and reshape
        7. Rotary embedding
        """
        # O projection
        hidden_states = self.o_proj(attn_output.flatten(1, -1))

        # Post-attention of current layer
        hidden_states, residual = self.post_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        # Pre-attention of next layer
        hidden_states, residual = self.next_input_norm(hidden_states, residual)

        # QKV projection
        qkv = self.next_qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Rotary embedding
        q, k = self.next_rotary_emb(positions, q, k)

        return q, k, v, residual

    def capture_graph(self, graph_pool=None):
        """Capture CUDA Graph."""
        # Allocate placeholders
        self.attn_in = torch.zeros(self.seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.r_in = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")
        self.pos_in = torch.zeros(self.seq_len, dtype=torch.long, device="cuda")

        self.q_out = torch.zeros(self.seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.k_out = torch.zeros(self.seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.v_out = torch.zeros(self.seq_len, self.num_kv_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.r_out = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                q, k, v, r = self._compute(self.attn_in, self.r_in, self.pos_in)
            self.q_out.copy_(q)
            self.k_out.copy_(k)
            self.v_out.copy_(v)
            self.r_out.copy_(r)
            torch.cuda.synchronize()

            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=graph_pool):
                q, k, v, r = self._compute(self.attn_in, self.r_in, self.pos_in)
                self.q_out.copy_(q)
                self.k_out.copy_(k)
                self.v_out.copy_(v)
                self.r_out.copy_(r)

        return self.graph.pool() if graph_pool is None else graph_pool

    def forward(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        use_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_graph and self.graph is not None and attn_output.shape[0] == self.seq_len:
            self.attn_in.copy_(attn_output)
            self.r_in.copy_(residual)
            self.pos_in.copy_(positions)
            self.graph.replay()
            return self.q_out.clone(), self.k_out.clone(), self.v_out.clone(), self.r_out.clone()
        else:
            return self._compute(attn_output, residual, positions)


class LastGraph(nn.Module):
    """
    Graph wrapper for last layer:
    o_proj → post_norm → mlp → final_norm

    Input: attn_output [seq_len, num_heads, head_dim], residual [seq_len, hidden_size]
    Output: hidden_states [seq_len, hidden_size]
    """

    def __init__(
        self,
        o_proj: nn.Module,
        post_norm: nn.Module,
        mlp: nn.Module,
        final_norm: nn.Module,
        # Shape parameters
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.o_proj = o_proj
        self.post_norm = post_norm
        self.mlp = mlp
        self.final_norm = final_norm

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Last layer computation:
        1. O projection
        2. Post-attention layernorm + residual
        3. MLP
        4. Final model norm + residual
        """
        hidden_states = self.o_proj(attn_output.flatten(1, -1))
        hidden_states, residual = self.post_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states

    def capture_graph(self, graph_pool=None):
        """Capture CUDA Graph."""
        # Allocate placeholders
        self.attn_in = torch.zeros(self.seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device="cuda")
        self.r_in = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")
        self.h_out = torch.zeros(self.seq_len, self.hidden_size, dtype=self.dtype, device="cuda")

        with torch.inference_mode():
            # Warmup
            for _ in range(3):
                h = self._compute(self.attn_in, self.r_in)
            self.h_out.copy_(h)
            torch.cuda.synchronize()

            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=graph_pool):
                h = self._compute(self.attn_in, self.r_in)
                self.h_out.copy_(h)

        return self.graph.pool() if graph_pool is None else graph_pool

    def forward(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        use_graph: bool = False,
    ) -> torch.Tensor:
        if use_graph and self.graph is not None and attn_output.shape[0] == self.seq_len:
            self.attn_in.copy_(attn_output)
            self.r_in.copy_(residual)
            self.graph.replay()
            return self.h_out.clone()
        else:
            return self._compute(attn_output, residual)


class OffloadGraphManager:
    """
    Manager for all CUDA Graphs in offload path.

    Creates and manages N+2 graphs for N-layer model:
    - 1 EmbedGraph
    - 1 FirstGraph
    - N-1 InterGraphs
    - 1 LastGraph

    Supports both Prefill and Decode modes via seq_len parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        """
        Initialize graph manager from model.

        Args:
            model: The CausalLM model (e.g., LlamaForCausalLM)
            seq_len: Sequence length (1 for decode, chunk_size for prefill)
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            dtype: Data type for tensors
        """
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Access model layers
        layers = model.model.layers
        num_layers = len(layers)
        self.num_layers = num_layers

        # Create EmbedGraph
        self.embed_graph = EmbedGraph(
            embed_tokens=model.model.embed_tokens,
            seq_len=seq_len,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        # Create FirstGraph: input_norm_0 → qkv_proj_0 → rotary_0
        self.first_graph = FirstGraph(
            input_norm=layers[0].input_layernorm,
            qkv_proj=layers[0].self_attn.qkv_proj,
            rotary_emb=layers[0].self_attn.rotary_emb,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        # Create InterGraphs: o_proj_i → post_norm_i → mlp_i → input_norm_{i+1} → qkv_proj_{i+1} → rotary_{i+1}
        self.inter_graphs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.inter_graphs.append(InterGraph(
                o_proj=layers[i].self_attn.o_proj,
                post_norm=layers[i].post_attention_layernorm,
                mlp=layers[i].mlp,
                next_input_norm=layers[i + 1].input_layernorm,
                next_qkv_proj=layers[i + 1].self_attn.qkv_proj,
                next_rotary_emb=layers[i + 1].self_attn.rotary_emb,
                seq_len=seq_len,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            ))

        # Create LastGraph: o_proj_{N-1} → post_norm_{N-1} → mlp_{N-1} → final_norm
        self.last_graph = LastGraph(
            o_proj=layers[-1].self_attn.o_proj,
            post_norm=layers[-1].post_attention_layernorm,
            mlp=layers[-1].mlp,
            final_norm=model.model.norm,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        self.captured = False
        self.graph_pool = None

    def capture_all(self):
        """Capture all graphs, sharing memory pool."""
        graph_pool = None

        # Capture embed graph
        graph_pool = self.embed_graph.capture_graph(graph_pool)

        # Capture first graph
        graph_pool = self.first_graph.capture_graph(graph_pool)

        # Capture inter-layer graphs
        for inter_graph in self.inter_graphs:
            graph_pool = inter_graph.capture_graph(graph_pool)

        # Capture last graph
        graph_pool = self.last_graph.capture_graph(graph_pool)

        self.graph_pool = graph_pool
        self.captured = True

    @property
    def num_graphs(self) -> int:
        """Total number of graphs: 1 + 1 + (N-1) + 1 = N+2"""
        return 1 + 1 + len(self.inter_graphs) + 1


# Legacy compatibility aliases (for gradual migration)
FirstLayerGraph = FirstGraph
InterLayerGraph = InterGraph
LastLayerGraph = LastGraph
