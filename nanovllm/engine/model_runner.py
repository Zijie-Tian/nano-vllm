import os
import pickle
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models import get_model_class
from nanovllm.layers.sampler import GreedySampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import get_logger
from nanovllm.kvcache import create_kvcache_manager, KVCacheManager

logger = get_logger("model_runner")


def _find_free_port() -> int:
    """Find a free port for distributed communication.

    Uses socket binding with port 0 to let the OS assign an available port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Dynamic port allocation: use env var if set, otherwise find a free port
        env_port = os.environ.get("NANOVLLM_DIST_PORT")
        if env_port is not None:
            port = int(env_port)
        else:
            port = _find_free_port()
            logger.info(f"Auto-assigned distributed port: {port}")
        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        model_class = get_model_class(hf_config)
        self.model = model_class(hf_config)
        load_model(self.model, config.model)
        self.sampler = GreedySampler()
        
        #> Disable warmup for debugging
        self.warmup_model()
        
        self.allocate_kv_cache()
        if not self.enforce_eager:
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
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
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
        context = get_context()
        # Use eager mode for: prefill, enforce_eager, large batch, or chunked attention
        # Chunked attention requires dynamic KV loading that can't be captured in CUDA Graph
        use_eager = is_prefill or self.enforce_eager or input_ids.size(0) > 512 or context.is_chunked_prefill
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
        #> Check if Chunked Offload mode should be used (all blocks on CPU)
        if hasattr(self, 'kvcache_manager') and hasattr(self.kvcache_manager, 'get_all_cpu_blocks'):
            use_chunked_offload = self._should_use_chunked_offload(seqs, is_prefill)
            if use_chunked_offload:
                if is_prefill:
                    return self.run_chunked_offload_prefill(seqs)
                else:
                    return self.run_chunked_offload_decode(seqs)

        #> Following Code will not use Chunked Offload mode
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _should_use_chunked_offload(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        """
        Check if three-region mode should be used.

        Use three-region when:
        - CPU offload is enabled
        - There are blocks on CPU (either allocated there or offloaded)
        - Sequence exceeds GPU Compute region capacity
        """
        if not hasattr(self.kvcache_manager, 'offload_engine'):
            return False

        for seq in seqs:
            if not seq.block_table:
                continue  # Skip warmup sequences

            # Check if any blocks are on CPU
            cpu_blocks, _ = self.kvcache_manager.get_all_cpu_blocks(seq)
            if cpu_blocks:
                # Has CPU blocks - use three-region
                return True

            # Check if sequence needs more blocks than GPU Compute region can hold
            compute_size = self.kvcache_manager.offload_engine.num_compute_blocks
            if seq.num_blocks > compute_size:
                # Needs chunked processing
                return True

        return False

    def run_chunked_offload_prefill(self, seqs: list[Sequence]) -> list[int]:
        """
        Run prefill with unified ring buffer (CPU is primary storage).

        Flow:
        1. All blocks are allocated to CPU (primary storage)
        2. Each chunk writes KV to ring buffer slot[chunk_idx % N]
        3. After each chunk, offload from ring buffer slot to CPU
        4. All N-1 other slots are used to load previous chunks for attention
        """
        assert len(seqs) == 1, "Ring buffer prefill only supports single sequence"
        seq = seqs[0]

        offload_engine = self.kvcache_manager.offload_engine
        # Each chunk uses 1 ring buffer slot = 1 block
        tokens_per_chunk = self.block_size

        total_tokens = len(seq)
        num_chunks = (total_tokens + tokens_per_chunk - 1) // tokens_per_chunk
        logger.debug(f"[Ring Buffer Prefill] Starting: {total_tokens} tokens, "
              f"ring_slots={offload_engine.num_ring_slots}, chunk={tokens_per_chunk} tokens, "
              f"total_chunks={num_chunks}")

        chunk_idx = 0
        logits = None
        processed_tokens = 0

        # Get CPU block table for offload targets
        cpu_block_ids, logical_ids = self.kvcache_manager.get_all_cpu_blocks(seq)

        while processed_tokens < total_tokens:
            chunk_start = processed_tokens
            chunk_end = min(processed_tokens + tokens_per_chunk, total_tokens)

            # Get ring buffer slot for this chunk
            write_slot = offload_engine.get_write_slot_for_prefill(chunk_idx)

            # CPU block index for this chunk
            block_idx = chunk_idx

            logger.debug(f"[Ring Buffer Prefill] Chunk {chunk_idx}: tokens {chunk_start}-{chunk_end}, "
                  f"write_slot={write_slot}")

            # Prepare inputs
            input_ids, positions = self._prepare_chunked_offload_chunk(
                seq, chunk_start, chunk_end, write_slot, block_idx, chunk_idx
            )

            if input_ids.numel() == 0:
                break

            #> Run model forward
            logits = self.run_model(input_ids, positions, is_prefill=True)
            reset_context()

            # Mark block as prefilled
            if block_idx < len(seq.block_table):
                logical_id = seq.block_table[block_idx]
                self.kvcache_manager.prefilled_blocks.add(logical_id)

            # NOTE: Per-layer async offloading is now done in attention.forward
            # Each layer offloads from its own prefill buffer - no waiting required!
            # The sparse policy hook is called in offload_prefill_buffer_async.

            processed_tokens = chunk_end
            chunk_idx += 1

        # Wait for all async prefill offloads to complete
        offload_engine.wait_all_prefill_offloads()

        logger.debug(f"[Ring Buffer Prefill] Complete: {chunk_idx} chunks")

        # Sample from last logits
        # For chunked prefill, ParallelLMHead automatically selects last position's logits
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if logits is not None:
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        else:
            token_ids = [0] if self.rank == 0 else None

        return token_ids

    def _prepare_chunked_offload_chunk(
        self,
        seq: Sequence,
        chunk_start: int,
        chunk_end: int,
        write_slot: int,
        block_idx: int,
        chunk_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs for a chunked offload prefill chunk (ring buffer design)."""
        # Input tokens for this chunk
        input_ids = seq[chunk_start:chunk_end]
        positions = list(range(chunk_start, chunk_end))

        # Create slot mapping pointing to the single write_slot
        slot_mapping = []
        for pos in range(chunk_start, chunk_end):
            pos_in_block = pos % self.block_size
            slot = write_slot * self.block_size + pos_in_block
            slot_mapping.append(slot)

        # Convert to tensors
        num_tokens = chunk_end - chunk_start
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Set up context for chunked prefill
        seqlen = num_tokens
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            slot_mapping=slot_mapping,
            is_chunked_prefill=True,
            kvcache_manager=self.kvcache_manager,
            chunked_seq=seq,
            current_chunk_idx=chunk_idx,  # Pass chunk index for ring buffer pipeline
        )

        return input_ids, positions

    def run_chunked_offload_decode(self, seqs: list[Sequence]) -> list[int]:
        """
        Run decode with cross-layer pipeline (CPU is primary storage).

        All KV is on CPU. Uses decode_slot (slot[0]) to write new KV.
        Optimized with cross-layer pipeline: Layer N's data is loaded while
        Layer N-1 computes, achieving transfer/compute overlap.

        Key: decode_slot is dedicated to writing new KV, never used for loading.
        Optimization: Cross-layer pipeline reduces effective latency by overlapping
        H2D transfers with attention computation across layers.
        """
        assert len(seqs) == 1, "Ring buffer decode only supports single sequence"
        seq = seqs[0]

        offload_engine = self.kvcache_manager.offload_engine

        # Prepare inputs
        input_ids = torch.tensor([seq.last_token], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([len(seq) - 1], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        # Use Decode region (slot 0) to write new KV
        decode_slot = offload_engine.decode_slot  # = 0
        pos_in_block = (len(seq) - 1) % self.block_size
        slot = decode_slot * self.block_size + pos_in_block
        slot_mapping = torch.tensor([slot], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_len = torch.tensor([len(seq)], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Get decode start position for accumulated token tracking
        decode_start_pos = self.kvcache_manager.get_decode_start_pos(seq)

        # Get prefilled CPU blocks for pipeline initialization
        cpu_block_table = self.kvcache_manager.get_prefilled_cpu_blocks(seq)

        # Start cross-layer pipeline (preloads Layer 0's data)
        offload_engine.start_decode_pipeline(cpu_block_table)

        # Set up context for chunked decode
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_len,
            is_chunked_prefill=True,  # Use chunked attention path
            kvcache_manager=self.kvcache_manager,
            chunked_seq=seq,
            decode_pos_in_block=pos_in_block,
            decode_start_pos_in_block=decode_start_pos,
        )

        # Run model forward pass
        logits = self.run_model(input_ids, positions, is_prefill=False)
        reset_context()

        # End cross-layer pipeline
        offload_engine.end_decode_pipeline()

        # Only offload when block is full (pos_in_block == block_size - 1)
        # This avoids unnecessary offloading on every decode step
        if pos_in_block == self.block_size - 1:
            last_cpu_block = self.kvcache_manager.get_last_cpu_block(seq)
            if last_cpu_block >= 0:
                # TODO: In new GPU cache architecture (no layer dimension),
                # decode offload should be done per-layer in attention.forward.
                # For now, offload all layers sequentially.
                for layer_id in range(offload_engine.num_layers):
                    offload_engine.offload_decode_slot_layer(layer_id, last_cpu_block)
                offload_engine.wait_all_offload_done()
            # Reset decode start position for next block
            self.kvcache_manager.reset_decode_start_pos(seq)

        # Sample
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
