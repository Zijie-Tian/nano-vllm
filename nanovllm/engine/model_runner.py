import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.kvcache import create_kvcache_manager, KVCacheManager


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
        self.sampler = Sampler()
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
        torch.cuda.synchronize()
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
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
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
            # Calculate CPU blocks based on cpu_memory_gb
            cpu_bytes = int(config.cpu_memory_gb * 1024**3)
            num_cpu_blocks = cpu_bytes // block_bytes
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

        # Bind layer caches to attention modules and set layer_id
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
        # Check if chunked prefill is needed
        if is_prefill and hasattr(self, 'kvcache_manager'):
            needs_chunked = any(
                hasattr(self.kvcache_manager, 'needs_chunked_prefill') and
                self.kvcache_manager.needs_chunked_prefill(seq)
                for seq in seqs if seq.block_table
            )
            if needs_chunked:
                return self.run_chunked_prefill(seqs)

        # Check if chunked decode is needed
        if not is_prefill and hasattr(self, 'kvcache_manager'):
            needs_chunked = any(
                hasattr(self.kvcache_manager, 'needs_chunked_decode') and
                self.kvcache_manager.needs_chunked_decode(seq)
                for seq in seqs if seq.block_table
            )
            if needs_chunked:
                return self.run_chunked_decode(seqs)

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def run_chunked_prefill(self, seqs: list[Sequence]) -> list[int]:
        """
        Run prefill in chunks when sequences exceed GPU capacity.

        For each chunk:
        1. Process tokens through model forward pass
        2. At each attention layer:
           - Load previous KV from CPU (handled by attention layer)
           - Compute attention with online softmax merging
           - Store current KV to GPU cache
        3. After chunk completes, offload KV to CPU
        4. Load next chunk's blocks to GPU
        """
        import sys

        # Currently only supporting single sequence for chunked prefill
        assert len(seqs) == 1, "Chunked prefill only supports single sequence"
        seq = seqs[0]

        total_blocks = seq.num_blocks
        print(f"[Chunked Prefill] Starting: {total_blocks} total blocks, "
              f"GPU slots: {self.kvcache_manager.num_gpu_slots}", file=sys.stderr)

        chunk_num = 0
        logits = None

        while True:
            # Get chunk info (which blocks are on GPU and not yet prefilled)
            chunk_info = self.kvcache_manager.get_gpu_block_tables_partial(seqs)
            gpu_blocks, start_block_idx, end_block_idx = chunk_info[0]

            if not gpu_blocks:
                # No more blocks to process
                break

            chunk_num += 1
            chunk_tokens = (end_block_idx - start_block_idx) * self.block_size
            if end_block_idx == seq.num_blocks:
                # Last block may be partial
                chunk_tokens = len(seq) - start_block_idx * self.block_size

            print(f"[Chunked Prefill] Chunk {chunk_num}: blocks {start_block_idx}-{end_block_idx-1}, "
                  f"~{chunk_tokens} tokens", file=sys.stderr)

            # Prepare inputs for this chunk
            input_ids, positions = self._prepare_chunked_prefill(seq, gpu_blocks, start_block_idx, end_block_idx)

            if input_ids.numel() == 0:
                print(f"[Chunked Prefill] No input tokens, breaking", file=sys.stderr)
                break

            print(f"[Chunked Prefill] Running model with {input_ids.numel()} tokens...", file=sys.stderr)

            # Run model forward pass
            logits = self.run_model(input_ids, positions, is_prefill=True)
            reset_context()

            print(f"[Chunked Prefill] Model forward complete", file=sys.stderr)

            # Check if this is the last chunk
            # Mark current chunk as prefilled and offload to CPU
            self.kvcache_manager.complete_prefill_chunk(seq)

            # Check if more chunks needed
            if not self.kvcache_manager.needs_chunked_prefill(seq):
                print(f"[Chunked Prefill] All chunks done, sampling", file=sys.stderr)
                break

            print(f"[Chunked Prefill] Chunk transfer complete, loading next...", file=sys.stderr)

        # Sample from the last chunk's logits
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if logits is not None:
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        else:
            token_ids = [0] if self.rank == 0 else None

        return token_ids

    def run_chunked_decode(self, seqs: list[Sequence]) -> list[int]:
        """
        Run decode with chunked attention when sequence exceeds GPU capacity.

        For decode, we need attention over ALL previous tokens. With CPU offload,
        we load KV chunks and compute attention incrementally per-layer.

        Flow:
        1. Ensure last block is on GPU (for writing new KV token)
        2. Run model forward - each attention layer:
           a. Compute attention on GPU blocks
           b. Load CPU blocks in chunks, compute + merge
        3. Sample from output
        """
        # Currently only supporting single sequence for chunked decode
        assert len(seqs) == 1, "Chunked decode only supports single sequence"
        seq = seqs[0]

        # Prepare inputs
        input_ids = torch.tensor([seq.last_token], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([len(seq) - 1], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        # Ensure last block is on GPU for writing new KV token
        last_gpu_slot = self.kvcache_manager.ensure_last_block_on_gpu(seq)
        slot = last_gpu_slot * self.block_size + seq.last_block_num_tokens - 1
        slot_mapping = torch.tensor([slot], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_len = torch.tensor([len(seq)], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Set up context for chunked decode
        set_context(
            is_prefill=False,  # Decode mode
            slot_mapping=slot_mapping,
            context_lens=context_len,
            is_chunked_prefill=True,  # Use chunked attention path
            offload_engine=self.kvcache_manager,
            chunked_seq=seq,
        )

        # Run model forward pass
        # Each attention layer will handle chunked KV loading internally
        logits = self.run_model(input_ids, positions, is_prefill=False)
        reset_context()

        # Sample
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        return token_ids

    def _prepare_chunked_prefill(
        self,
        seq: Sequence,
        gpu_blocks: list[int],
        start_block_idx: int,
        end_block_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for a single chunk in chunked prefill.

        Sets up context with is_chunked_prefill=True so attention layers
        know to load previous KV from CPU.
        """
        # Calculate token range for this chunk
        start_token = start_block_idx * self.block_size
        end_token = min(end_block_idx * self.block_size, len(seq))

        # Input tokens for this chunk
        input_ids = seq[start_token:end_token]
        positions = list(range(start_token, end_token))

        # Slot mapping for storing KV cache
        slot_mapping = []
        for i, gpu_block_id in enumerate(gpu_blocks):
            block_idx = start_block_idx + i
            start = gpu_block_id * self.block_size
            if block_idx != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))

        # Trim slot_mapping to match actual token count
        actual_tokens = end_token - start_token
        slot_mapping = slot_mapping[:actual_tokens]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Set up context for chunked prefill
        seqlen = actual_tokens
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
            offload_engine=self.kvcache_manager,  # Pass manager for loading previous KV
            chunked_seq=seq,  # Pass sequence for loading previous KV
        )

        return input_ids, positions

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
