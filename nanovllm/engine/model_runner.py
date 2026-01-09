import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config, SparsePolicyType
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import GreedySampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import get_logger
from nanovllm.kvcache import create_kvcache_manager, KVCacheManager

logger = get_logger("model_runner")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = GreedySampler()

        # Initialize sparse_prefill_policy before warmup (will be configured in allocate_kv_cache)
        self.sparse_prefill_policy = None

        #> Disable warmup for debugging
        self.warmup_model()
        
        self.allocate_kv_cache()
        if not self.enforce_eager:
            if config.enable_cpu_offload:
                # TODO: Implement capture_offload_cudagraph() for offload mode
                # For now, offload mode uses eager execution
                # The standard capture_cudagraph() cannot be used because:
                # - It captures the PagedAttention decode path via Attention.forward()
                # - In offload mode, Attention.k_cache/v_cache are empty (KV is in ring buffer)
                # - The refactored offload decode now uses Attention.forward() with ring buffer
                # - Need specialized graph capture that sets up ring buffer correctly
                pass
            else:
                self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Use a reasonable warmup length instead of max_model_len
        # Warmup only needs to trigger CUDA kernel JIT compilation
        # Using 2 blocks is sufficient and avoids huge memory allocation
        warmup_len = min(self.block_size * 2, self.config.max_model_len)
        warmup_len = max(warmup_len, 128)  # At least 128 tokens
        num_seqs = min(self.config.max_num_batched_tokens // warmup_len, self.config.max_num_seqs, 4)
        num_seqs = max(num_seqs, 1)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

        # Calculate max GPU blocks based on available memory
        max_gpu_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert max_gpu_blocks > 0

        # Determine final GPU blocks: user-specified or auto (max available)
        if config.num_gpu_blocks > 0:
            num_gpu_blocks = min(config.num_gpu_blocks, max_gpu_blocks)
        else:
            num_gpu_blocks = max_gpu_blocks

        if config.enable_cpu_offload:
            # Three-region design: CPU is primary storage, GPU is working buffer
            # CPU blocks = all blocks needed to support max_model_len (stores complete KV for one max sequence)
            # GPU blocks = three-region working buffer (user-specified or auto)
            num_cpu_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

            config.num_gpu_kvcache_blocks = num_gpu_blocks
            config.num_cpu_kvcache_blocks = num_cpu_blocks
            # For backward compatibility
            config.num_kvcache_blocks = num_gpu_blocks + num_cpu_blocks
        else:
            config.num_kvcache_blocks = num_gpu_blocks
            config.num_gpu_kvcache_blocks = num_gpu_blocks
            config.num_cpu_kvcache_blocks = 0

        # Create KV cache manager using factory
        self.kvcache_manager: KVCacheManager = create_kvcache_manager(config)

        # Create sparse prefill policy for GPU-only path
        # This is separate from CPU offload sparse policy (which uses select_blocks)
        self.sparse_prefill_policy = None
        if not config.enable_cpu_offload and config.sparse_policy != SparsePolicyType.FULL:
            from nanovllm.kvcache.sparse import create_sparse_policy
            policy = create_sparse_policy(
                config.sparse_policy,
                vertical_size=config.minference_vertical_size,
                slash_size=config.minference_slash_size,
                adaptive_budget=config.minference_adaptive_budget,
                num_sink_tokens=config.minference_num_sink_tokens,
                num_recent_diags=config.minference_num_recent_diags,
            )
            # Only use if policy supports sparse prefill
            if policy.supports_prefill:
                self.sparse_prefill_policy = policy
                logger.info(f"Sparse prefill policy enabled: {self.sparse_prefill_policy}")

        # Allocate cache through manager
        self.kvcache_manager.allocate_cache(
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=hf_config.torch_dtype,
        )

        # Initialize sparse policy if manager has one (CPU offload mode)
        if hasattr(self.kvcache_manager, 'sparse_policy') and self.kvcache_manager.sparse_policy is not None:
            self.kvcache_manager.sparse_policy.initialize(
                num_layers=hf_config.num_hidden_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_cpu_blocks=config.num_cpu_kvcache_blocks,
                dtype=hf_config.torch_dtype,
                device=torch.device("cuda"),
            )

            logger.info(
                f"Sparse policy initialized: {config.sparse_policy.name} "
                f"(topk={config.sparse_topk_blocks}, threshold={config.sparse_threshold_blocks})"
            )

        # Log KV cache allocation info with detailed per-token breakdown
        gpu_memory_mb = config.num_gpu_kvcache_blocks * block_bytes / (1024 ** 2)
        cpu_memory_mb = config.num_cpu_kvcache_blocks * block_bytes / (1024 ** 2)
        total_memory_mb = gpu_memory_mb + cpu_memory_mb

        # Calculate per-token KV cache usage
        # KV per token = 2 (K+V) * num_layers * kv_heads * head_dim * dtype_size
        dtype_size = 2 if hf_config.torch_dtype in [torch.float16, torch.bfloat16] else 4
        per_token_kv_bytes = 2 * hf_config.num_hidden_layers * num_kv_heads * head_dim * dtype_size
        per_token_kv_kb = per_token_kv_bytes / 1024

        logger.info(
            f"KV Cache per-token: {per_token_kv_kb:.2f}KB "
            f"(2 * {hf_config.num_hidden_layers}layers * {num_kv_heads}kv_heads * {head_dim}head_dim * {dtype_size}bytes)"
        )
        logger.info(
            f"KV Cache per-block: {block_bytes / (1024**2):.2f}MB "
            f"({per_token_kv_kb:.2f}KB * {self.block_size}tokens)"
        )

        if config.enable_cpu_offload:
            compute_size = config.num_gpu_kvcache_blocks // 2
            tokens_per_chunk = compute_size * self.block_size
            logger.info(
                f"KV Cache allocated (Chunked Offload mode): "
                f"GPU={config.num_gpu_kvcache_blocks} blocks ({gpu_memory_mb:.1f}MB), "
                f"CPU={config.num_cpu_kvcache_blocks} blocks ({cpu_memory_mb:.1f}MB), "
                f"Total={total_memory_mb:.1f}MB"
            )
            logger.info(
                f"Chunked Offload config: compute_size={compute_size} blocks, "
                f"tokens_per_chunk={tokens_per_chunk}, "
                f"block_size={self.block_size}"
            )
        else:
            logger.info(
                f"KV Cache allocated: "
                f"GPU={config.num_gpu_kvcache_blocks} blocks ({gpu_memory_mb:.1f}MB), "
                f"block_size={self.block_size}"
            )

        #> Bind layer caches to attention modules and set layer_id
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                k_cache, v_cache = self.kvcache_manager.get_layer_cache(layer_id)
                module.k_cache = k_cache
                module.v_cache = v_cache
                # Set layer_id for chunked prefill support
                if hasattr(module, "layer_id"):
                    module.layer_id = layer_id
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], chunk_info: list[tuple] = None):
        """
        Prepare inputs for prefill.

        Args:
            seqs: List of sequences to prefill
            chunk_info: Optional chunked prefill info from get_gpu_block_tables_partial().
                       If provided, only process blocks in the chunk.
                       Format: [(gpu_block_ids, start_block_idx, end_block_idx), ...]
        """
        # Check if any sequence has blocks (not warmup)
        has_blocks = any(seq.block_table for seq in seqs)

        gpu_block_tables = None
        if has_blocks and hasattr(self, 'kvcache_manager'):
            if chunk_info is None:
                # Standard prefill - try to get all blocks
                # This may fail if GPU doesn't have enough capacity
                self.kvcache_manager.prepare_for_attention(seqs, is_prefill=True)
                gpu_block_tables = self.kvcache_manager.get_gpu_block_tables(seqs)
            else:
                # Chunked prefill - use provided chunk info
                gpu_block_tables = [info[0] for info in chunk_info]

        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq_idx, seq in enumerate(seqs):
            if chunk_info is not None:
                # Chunked prefill: only process blocks in the chunk
                gpu_blocks, start_block_idx, end_block_idx = chunk_info[seq_idx]
                if not gpu_blocks:
                    continue

                # Calculate token range for this chunk
                start_token = start_block_idx * self.block_size
                end_token = min(end_block_idx * self.block_size, len(seq))
                if end_block_idx == seq.num_blocks:
                    # Last chunk includes partial last block
                    end_token = len(seq)

                # Input tokens for this chunk
                chunk_tokens = seq[start_token:end_token]
                input_ids.extend(chunk_tokens)
                positions.extend(list(range(start_token, end_token)))

                seqlen_q = end_token - start_token
                seqlen_k = end_token  # Context includes all tokens up to this point
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                # Slot mapping for blocks in this chunk
                for i, gpu_block_id in enumerate(gpu_blocks):
                    block_idx = start_block_idx + i
                    start = gpu_block_id * self.block_size
                    if block_idx != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    slot_mapping.extend(list(range(start, end)))
            else:
                # Standard prefill
                seqlen = len(seq)
                input_ids.extend(seq[seq.num_cached_tokens:])
                positions.extend(list(range(seq.num_cached_tokens, seqlen)))
                seqlen_q = seqlen - seq.num_cached_tokens
                seqlen_k = seqlen
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)
                if not seq.block_table:    # warmup
                    continue
                # Use GPU physical block IDs for slot mapping
                gpu_blocks = gpu_block_tables[seq_idx]
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = gpu_blocks[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1] and gpu_block_tables:    # prefix cache
            block_tables = self._prepare_gpu_block_tables(gpu_block_tables)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables,
                    sparse_prefill_policy=self.sparse_prefill_policy)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # Prepare KV cache (updates gather_indices for hybrid manager)
        if hasattr(self, 'kvcache_manager'):
            self.kvcache_manager.prepare_for_attention(seqs, is_prefill=False)
            # Get GPU physical block tables
            gpu_block_tables = self.kvcache_manager.get_gpu_block_tables(seqs)
        else:
            gpu_block_tables = [list(seq.block_table) for seq in seqs]

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq_idx, seq in enumerate(seqs):
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # Use GPU physical block ID for slot mapping
            gpu_blocks = gpu_block_tables[seq_idx]
            slot_mapping.append(gpu_blocks[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Use GPU physical block tables for attention
        block_tables = self._prepare_gpu_block_tables(gpu_block_tables)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def _prepare_gpu_block_tables(self, gpu_block_tables: list[list[int]]):
        """Prepare block tables tensor from GPU physical block IDs."""
        max_len = max(len(bt) for bt in gpu_block_tables)
        padded = [bt + [-1] * (max_len - len(bt)) for bt in gpu_block_tables]
        return torch.tensor(padded, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Use eager mode for: prefill, enforce_eager, large batch
        use_eager = is_prefill or self.enforce_eager or input_ids.size(0) > 512
        if use_eager:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        #> Check if Layer-wise Offload mode should be used (CPU offload enabled)
        if hasattr(self, 'kvcache_manager') and hasattr(self.kvcache_manager, 'offload_engine'):
            use_layerwise_offload = self._should_use_layerwise_offload(seqs, is_prefill)
            if use_layerwise_offload:
                if is_prefill:
                    return self.run_layerwise_offload_prefill(seqs)
                else:
                    return self.run_layerwise_offload_decode(seqs)

        #> Check if contiguous GPU mode should be used (single-seq optimization)
        if self._should_use_contiguous_gpu_mode(seqs, is_prefill):
            if is_prefill:
                return self.run_gpu_only_prefill(seqs)
            else:
                return self.run_gpu_only_decode(seqs)

        #> Following Code uses standard PagedAttention path
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _should_use_contiguous_gpu_mode(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        """
        Check if contiguous GPU mode should be used for single-seq optimization.

        Conditions:
        1. Has kvcache_manager with contiguous cache allocated
        2. Not using CPU offload (no offload_engine)
        3. Single sequence (batch_size == 1)
        4. Has blocks allocated (not warmup)
        """
        # Must have kvcache_manager
        if not hasattr(self, 'kvcache_manager') or self.kvcache_manager is None:
            return False

        # Must have contiguous cache
        if not hasattr(self.kvcache_manager, 'contiguous_k_cache'):
            return False
        if self.kvcache_manager.contiguous_k_cache is None:
            return False

        # Must NOT be offload mode
        if hasattr(self.kvcache_manager, 'offload_engine'):
            return False

        # Single sequence only
        if len(seqs) != 1:
            return False

        # Has blocks allocated (not warmup)
        if not seqs[0].block_table:
            return False

        return True

    # ========== Contiguous GPU-only Methods ==========

    @torch.inference_mode()
    def run_gpu_only_prefill(self, seqs: list[Sequence]) -> list[int]:
        """
        GPU-only prefill with contiguous KV cache layout.

        Mirrors run_layerwise_offload_prefill() but stores to GPU instead of CPU.
        No scatter operations - just contiguous slice assignment.

        Key design:
        - Process layer-by-layer (not via Attention.forward())
        - Store K,V to contiguous GPU cache (same layout as computed K,V)
        - Use sparse prefill attention if enabled
        """
        assert len(seqs) == 1, "GPU-only layer-wise prefill only supports single sequence"
        seq = seqs[0]

        num_layers = len(self.model.model.layers)
        total_tokens = len(seq)

        logger.debug(f"[GPU-only Prefill] Starting: {total_tokens} tokens, {num_layers} layers")

        # Get contiguous GPU cache
        k_cache = self.kvcache_manager.contiguous_k_cache
        v_cache = self.kvcache_manager.contiguous_v_cache

        # Prepare inputs
        input_ids = torch.tensor(seq[:], dtype=torch.int64, device="cuda")
        positions = torch.arange(total_tokens, dtype=torch.int64, device="cuda")

        # Import FlashAttention
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

        # Embedding
        hidden_states = self.model.model.embed_tokens(input_ids)
        residual = None

        # Layer-by-layer processing
        for layer_id in range(num_layers):
            layer = self.model.model.layers[layer_id]

            # Input LayerNorm
            if residual is None:
                hidden_ln, residual = layer.input_layernorm(hidden_states), hidden_states
            else:
                hidden_ln, residual = layer.input_layernorm(hidden_states, residual)

            # QKV projection
            qkv = layer.self_attn.qkv_proj(hidden_ln)
            q, k, v = qkv.split([
                layer.self_attn.q_size,
                layer.self_attn.kv_size,
                layer.self_attn.kv_size
            ], dim=-1)

            q = q.view(total_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k = k.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            v = v.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

            # Q/K norms (Qwen3 specific)
            if not layer.self_attn.qkv_bias:
                num_tokens = q.shape[0]
                q = layer.self_attn.q_norm(q.reshape(-1, layer.self_attn.head_dim))
                q = q.view(num_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
                k = layer.self_attn.k_norm(k.reshape(-1, layer.self_attn.head_dim))
                k = k.view(num_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

            # RoPE
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # Sparse or Full attention (uses k, v directly - before store!)
            if self.sparse_prefill_policy is not None:
                attn_output = self.sparse_prefill_policy.sparse_prefill_attention(
                    q, k, v, layer_id
                )
            else:
                attn_output = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=total_tokens,
                    max_seqlen_k=total_tokens,
                    softmax_scale=layer.self_attn.attn.scale,
                    causal=True,
                )

            # O projection
            attn_output = attn_output.view(total_tokens, -1)
            hidden_states = layer.self_attn.o_proj(attn_output)

            # Store K,V to contiguous GPU cache AFTER attention (same as offload pattern)
            k_cache[layer_id, :total_tokens] = k
            v_cache[layer_id, :total_tokens] = v

            # Post-attention LayerNorm + MLP
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)

        # Final norm
        hidden_states, _ = self.model.model.norm(hidden_states, residual)

        # Compute logits for last token
        logits = self.model.compute_logits(hidden_states[-1:])

        # Record prefill length for decode
        self.kvcache_manager.contiguous_seq_len = total_tokens

        logger.debug(f"[GPU-only Prefill] Complete: {num_layers} layers processed")

        # Sample
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        return token_ids

    @torch.inference_mode()
    def run_gpu_only_decode(self, seqs: list[Sequence]) -> list[int]:
        """
        Decode using contiguous GPU KV cache.

        Similar to offload decode but simpler - all KV already on GPU.
        """
        assert len(seqs) == 1, "GPU-only decode only supports single sequence"
        seq = seqs[0]

        num_layers = len(self.model.model.layers)
        k_cache = self.kvcache_manager.contiguous_k_cache
        v_cache = self.kvcache_manager.contiguous_v_cache
        context_len = self.kvcache_manager.contiguous_seq_len

        # Prepare inputs
        input_ids = torch.tensor([seq.last_token], dtype=torch.int64, device="cuda")
        positions = torch.tensor([len(seq) - 1], dtype=torch.int64, device="cuda")

        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

        # Embedding
        hidden_states = self.model.model.embed_tokens(input_ids)
        residual = None

        for layer_id in range(num_layers):
            layer = self.model.model.layers[layer_id]

            # Input LayerNorm
            if residual is None:
                hidden_ln, residual = layer.input_layernorm(hidden_states), hidden_states
            else:
                hidden_ln, residual = layer.input_layernorm(hidden_states, residual)

            # QKV projection
            qkv = layer.self_attn.qkv_proj(hidden_ln)
            q, k_new, v_new = qkv.split([
                layer.self_attn.q_size,
                layer.self_attn.kv_size,
                layer.self_attn.kv_size
            ], dim=-1)

            q = q.view(1, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k_new = k_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
            v_new = v_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

            # Q/K norms
            if not layer.self_attn.qkv_bias:
                q = layer.self_attn.q_norm(q.reshape(-1, layer.self_attn.head_dim))
                q = q.view(1, layer.self_attn.num_heads, layer.self_attn.head_dim)
                k_new = layer.self_attn.k_norm(k_new.reshape(-1, layer.self_attn.head_dim))
                k_new = k_new.view(1, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

            # RoPE
            q, k_new = layer.self_attn.rotary_emb(positions, q, k_new)

            # Store new K,V to cache
            k_cache[layer_id, context_len] = k_new.squeeze(0)
            v_cache[layer_id, context_len] = v_new.squeeze(0)

            # Full K,V for attention (including new token)
            k_full = k_cache[layer_id, :context_len + 1]
            v_full = v_cache[layer_id, :context_len + 1]

            # Attention
            cu_seqlens_k = torch.tensor([0, context_len + 1], dtype=torch.int32, device="cuda")
            attn_output = flash_attn_varlen_func(
                q, k_full, v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=context_len + 1,
                softmax_scale=layer.self_attn.attn.scale,
                causal=False,  # Single query, no causal needed
            )

            # O projection
            attn_output = attn_output.view(1, -1)
            hidden_states = layer.self_attn.o_proj(attn_output)

            # Post-attention LayerNorm + MLP
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)

        # Update context length
        self.kvcache_manager.contiguous_seq_len = context_len + 1

        # Final norm
        hidden_states, _ = self.model.model.norm(hidden_states, residual)

        # Compute logits
        logits = self.model.compute_logits(hidden_states)

        # Sample
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        return token_ids

    def _should_use_layerwise_offload(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        """
        Check if layer-wise offload mode should be used.

        Use layer-wise offload when:
        - CPU offload is enabled (offload_engine exists)
        - Sequence has blocks allocated (not warmup)
        """
        if not hasattr(self.kvcache_manager, 'offload_engine'):
            return False

        for seq in seqs:
            if seq.block_table:
                # Has blocks - use layer-wise offload
                return True

        return False

    # ========== Layer-wise Offload Methods ==========

    @torch.inference_mode()
    def run_layerwise_offload_prefill(self, seqs: list[Sequence]) -> list[int]:
        """
        Run prefill with layer-wise processing and async CPU offload.

        Key design:
        - Process one layer at a time (not one chunk at a time)
        - Each layer: compute → async offload KV to CPU
        - Offload of layer N overlaps with compute of layer N+1
        - Uses OffloadEngine's async API with stream events

        This enables future sparse attention methods (like MInference)
        that need full KV context per layer for pattern estimation.
        """
        assert len(seqs) == 1, "Layer-wise offload only supports single sequence"
        seq = seqs[0]

        offload_engine = self.kvcache_manager.offload_engine
        compute_stream = offload_engine.compute_stream
        num_layers = len(self.model.model.layers)
        total_tokens = len(seq)

        logger.debug(f"[Layer-wise Prefill] Starting: {total_tokens} tokens, {num_layers} layers")

        # Get CPU block IDs for offload targets
        cpu_block_ids, logical_ids = self.kvcache_manager.get_all_cpu_blocks(seq)

        # Prepare inputs
        input_ids = torch.tensor(seq[:], dtype=torch.int64, device="cuda")
        positions = torch.arange(total_tokens, dtype=torch.int64, device="cuda")

        # Import FlashAttention once
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device="cuda")

        # Step 1: Embedding (on compute stream)
        with torch.cuda.stream(compute_stream):
            hidden_states = self.model.model.embed_tokens(input_ids)
            residual = None

            # Step 2: Layer-by-layer processing
            for layer_id in range(num_layers):
                layer = self.model.model.layers[layer_id]

                # 2a. Input LayerNorm
                if residual is None:
                    hidden_ln, residual = layer.input_layernorm(hidden_states), hidden_states
                else:
                    hidden_ln, residual = layer.input_layernorm(hidden_states, residual)

                # 2b. Self-attention (full sequence)
                # QKV projection
                qkv = layer.self_attn.qkv_proj(hidden_ln)
                q, k, v = qkv.split([
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size
                ], dim=-1)

                q = q.view(total_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
                k = k.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                v = v.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

                # Q/K norms (Qwen3 specific)
                if not layer.self_attn.qkv_bias:
                    num_tokens = q.shape[0]
                    q = layer.self_attn.q_norm(q.reshape(-1, layer.self_attn.head_dim))
                    q = q.view(num_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
                    k = layer.self_attn.k_norm(k.reshape(-1, layer.self_attn.head_dim))
                    k = k.view(num_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)

                # RoPE
                q, k = layer.self_attn.rotary_emb(positions, q, k)

                # Sparse or Full attention
                if self.sparse_prefill_policy is not None:
                    # MInference or other sparse prefill policy
                    attn_output = self.sparse_prefill_policy.sparse_prefill_attention(
                        q, k, v, layer_id
                    )
                else:
                    # Full attention using FlashAttention
                    attn_output = flash_attn_varlen_func(
                        q, k, v,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=total_tokens,
                        max_seqlen_k=total_tokens,
                        softmax_scale=layer.self_attn.attn.scale,
                        causal=True,
                    )

                # O projection
                attn_output = attn_output.view(total_tokens, -1)
                hidden_states = layer.self_attn.o_proj(attn_output)

                # 2c. Post-attention LayerNorm + MLP
                hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                hidden_states = layer.mlp(hidden_states)

                # 2d. Offload KV to CPU (encapsulated with sparse policy hooks)
                offload_engine.offload_layer_kv_sync(layer_id, k, v, cpu_block_ids, total_tokens)

            # Step 3: Final norm
            hidden_states, _ = self.model.model.norm(hidden_states, residual)

            # Step 4: Compute logits for last token
            logits = self.model.compute_logits(hidden_states[-1:])

        # Note: Using sync offload, no wait needed

        # Mark all blocks as prefilled
        for logical_id in logical_ids:
            self.kvcache_manager.prefilled_blocks.add(logical_id)

        # Step 5: Sample
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        logger.debug(f"[Layer-wise Prefill] Complete: {num_layers} layers processed")

        return token_ids

    @torch.inference_mode()
    def run_layerwise_offload_decode(self, seqs: list[Sequence]) -> list[int]:
        """
        Run decode with ring-buffered layer-wise KV loading from CPU.

        Key design:
        - Ring buffer pipeline: load layer N+k while computing layer N
        - Uses standard Attention.forward() path (not bypassing)
        - Per-layer decode buffer for accumulating new tokens
        - Async block offload when decode buffer is full
        """
        assert len(seqs) == 1, "Layer-wise offload only supports single sequence"
        seq = seqs[0]

        offload_engine = self.kvcache_manager.offload_engine
        compute_stream = offload_engine.compute_stream
        num_layers = len(self.model.model.layers)
        num_buffers = offload_engine.num_kv_buffers

        # Prepare inputs
        input_ids = torch.tensor([seq.last_token], dtype=torch.int64, device="cuda")
        positions = torch.tensor([len(seq) - 1], dtype=torch.int64, device="cuda")

        # Get prefilled CPU blocks and compute valid tokens per block
        cpu_block_table = self.kvcache_manager.get_prefilled_cpu_blocks(seq)
        num_prefill_blocks = len(cpu_block_table)
        total_prefill_tokens = self.kvcache_manager.get_prefill_len(seq)

        # Calculate valid tokens per block
        valid_tokens_per_block = []
        for block_idx in range(num_prefill_blocks):
            if block_idx == num_prefill_blocks - 1:
                # Last block may be partial
                last_block_tokens = total_prefill_tokens % self.block_size
                if last_block_tokens == 0 and total_prefill_tokens > 0:
                    last_block_tokens = self.block_size
                valid_tokens_per_block.append(last_block_tokens)
            else:
                valid_tokens_per_block.append(self.block_size)

        # Current decode position info
        pos_in_block = (len(seq) - 1) % self.block_size
        decode_start_pos = self.kvcache_manager.get_decode_start_pos(seq)
        num_prev_decode_tokens = pos_in_block - decode_start_pos  # Previous decode tokens (not including current)

        # Total context length (prefill + previous decode tokens)
        # New token will be stored at this position
        context_len = total_prefill_tokens + num_prev_decode_tokens

        # Context setup for Attention.forward() - contiguous mode (no block tables)
        slot_mapping = torch.tensor([context_len], dtype=torch.int32, device="cuda")
        context_lens = torch.tensor([context_len + 1], dtype=torch.int32, device="cuda")

        # Phase 1: Preload first N layers to ring buffer (fill pipeline)
        num_preload = min(num_buffers, num_layers)
        for i in range(num_preload):
            offload_engine.load_layer_kv_to_buffer(
                i, i, cpu_block_table, valid_tokens_per_block
            )

        # Step 1: Embedding (on compute stream)
        with torch.cuda.stream(compute_stream):
            hidden_states = self.model.model.embed_tokens(input_ids)
            residual = None

            # Phase 2: Layer-by-layer processing with ring buffer pipeline
            for layer_id in range(num_layers):
                layer = self.model.model.layers[layer_id]
                attn_module = layer.self_attn.attn  # The Attention module
                current_buffer = layer_id % num_buffers

                # 2a. Wait for current buffer's load to complete
                offload_engine.wait_buffer_load(current_buffer)

                # 2b. Copy previous decode KV from decode buffer to ring buffer
                # Ring buffer already has prefill KV at [0:total_prefill_tokens]
                # We need to add decode KV at [total_prefill_tokens:]
                if num_prev_decode_tokens > 0:
                    k_decode_prev, v_decode_prev = offload_engine.get_decode_kv(
                        layer_id, decode_start_pos, pos_in_block
                    )
                    ring_k = offload_engine.layer_k_cache[current_buffer]
                    ring_v = offload_engine.layer_v_cache[current_buffer]
                    ring_k[total_prefill_tokens:total_prefill_tokens + num_prev_decode_tokens].copy_(k_decode_prev)
                    ring_v[total_prefill_tokens:total_prefill_tokens + num_prev_decode_tokens].copy_(v_decode_prev)

                # 2c. Set Attention module's cache to ring buffer (contiguous format)
                # Shape: [max_seq_len, kv_heads, head_dim] -> [1, max_seq_len, kv_heads, head_dim]
                attn_module.k_cache = offload_engine.layer_k_cache[current_buffer:current_buffer+1]
                attn_module.v_cache = offload_engine.layer_v_cache[current_buffer:current_buffer+1]

                # 2d. Set context for Attention.forward() - contiguous mode
                set_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=None,  # Contiguous mode, no block tables
                )

                # 2e. Forward through layer using standard path
                # This calls Qwen3Attention.forward() -> Attention.forward()
                # Attention.forward() will:
                #   - Store new K,V to ring buffer via store_kvcache
                #   - Compute attention via flash_attn_with_kvcache
                hidden_states, residual = layer(positions, hidden_states, residual)

                # 2f. Copy new token's KV from ring buffer to decode buffer (for persistence)
                # The new token was stored at position context_len in ring buffer
                ring_k = offload_engine.layer_k_cache[current_buffer]
                ring_v = offload_engine.layer_v_cache[current_buffer]
                offload_engine.decode_k_buffer[layer_id, pos_in_block].copy_(ring_k[context_len])
                offload_engine.decode_v_buffer[layer_id, pos_in_block].copy_(ring_v[context_len])

                # 2g. Mark buffer compute done (allows next load to reuse this buffer)
                offload_engine.record_buffer_compute_done(current_buffer)

                # 2h. Start loading next layer to same buffer (after compute done)
                next_layer_to_load = layer_id + num_buffers
                if next_layer_to_load < num_layers:
                    offload_engine.load_layer_kv_to_buffer(
                        current_buffer, next_layer_to_load, cpu_block_table, valid_tokens_per_block
                    )

            # Step 3: Final norm
            hidden_states, _ = self.model.model.norm(hidden_states, residual)

            # Step 4: Compute logits
            logits = self.model.compute_logits(hidden_states)

        # Reset context
        reset_context()

        # Step 5: Handle block-full offload (async)
        if pos_in_block == self.block_size - 1:
            last_cpu_block = self.kvcache_manager.get_last_cpu_block(seq)
            if last_cpu_block >= 0:
                # Async offload decode buffer to CPU
                offload_engine.offload_decode_buffer_async(last_cpu_block)

                # Mark as prefilled for future decode steps
                logical_id = seq.block_table[-1]
                self.kvcache_manager.prefilled_blocks.add(logical_id)

            # Reset decode start position
            self.kvcache_manager.reset_decode_start_pos(seq)

        # Step 6: Sample
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
