import atexit
from dataclasses import fields
from time import perf_counter, perf_counter_ns
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.observer import InferenceObserver
from nanovllm.utils.memory_observer import MemoryObserver


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        # Set Sequence.block_size to match the KV cache block size
        Sequence.block_size = config.kvcache_block_size
        self.scheduler = Scheduler(config, self.model_runner.kvcache_manager)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        import os
        debug_enabled = os.environ.get('NANOVLLM_LOG_LEVEL', 'INFO').upper() == 'DEBUG'

        seqs, is_prefill = self.scheduler.schedule()
        if debug_enabled:
            mode = "PREFILL" if is_prefill else "DECODE"
            print(f"[DEBUG LLMEngine.step] Mode={mode}, active_sequences={len(seqs)}")

        if not is_prefill:
            # Decode mode: calculate TPOT from previous decode step
            if InferenceObserver.tpot_start != 0:
                InferenceObserver.tpot = perf_counter_ns() - InferenceObserver.tpot_start
            InferenceObserver.tpot_start = perf_counter_ns()

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if is_prefill:
            # Calculate TTFT after prefill completes (including chunked prefill)
            if InferenceObserver.ttft_start != 0:
                InferenceObserver.ttft = perf_counter_ns() - InferenceObserver.ttft_start
                InferenceObserver.reset_ttft()
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        if debug_enabled and outputs:
            for seq_id, tokens in outputs:
                print(f"[DEBUG LLMEngine.step] Sequence {seq_id} finished, {len(tokens)} tokens generated")

        #> Calculate number of tokens processed
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        import os
        log_level = os.environ.get('NANOVLLM_LOG_LEVEL', 'INFO')
        debug_enabled = log_level.upper() == 'DEBUG'

        InferenceObserver.complete_reset()
        MemoryObserver.complete_reset()
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        iteration = 0
        last_output_count = 0

        while not self.is_finished():
            if debug_enabled and iteration % 100 == 0:
                print(f"[DEBUG LLMEngine] Iteration {iteration}, finished_sequences={len(outputs)}, total_prompts={len(prompts)}")

            # Timeout check (32K sample should finish within 20 minutes = 1200 seconds)
            if iteration == 0:
                import time
                start_time = time.time()
            elif debug_enabled and iteration % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed > 1200:  # 20 minutes
                    print(f"[WARNING] Test exceeded 20 minutes timeout! Iteration={iteration}, forcing exit.")
                    import sys
                    sys.exit(1)

            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                    "ttft": f"{float(InferenceObserver.ttft) / 1e6}ms",
                    "tpot": f"{float(InferenceObserver.tpot) / 1e6}ms",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
