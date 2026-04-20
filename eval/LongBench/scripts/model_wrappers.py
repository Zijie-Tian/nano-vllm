from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    dataset: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 32
    stop: Optional[list[str]] = None


def _resolve_dtype(dtype: str) -> torch.dtype:
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch, dtype)


def _trim_stop_strings(text: str, stop: Optional[Iterable[str]]) -> str:
    if not stop:
        return text
    for needle in stop:
        if needle:
            text = text.split(needle)[0]
    return text


def load_tokenizer_with_fallback(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    attempts = [
        {"trust_remote_code": trust_remote_code, "use_fast": False},
        {"trust_remote_code": trust_remote_code},
        {"trust_remote_code": trust_remote_code, "use_fast": True},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            return tokenizer
        last_error = TypeError(
            "AutoTokenizer.from_pretrained returned "
            f"{type(tokenizer).__name__} for {model_name_or_path} with kwargs={kwargs}"
        )
    raise RuntimeError(f"Failed to load a valid tokenizer from {model_name_or_path}") from last_error


def _resolve_eos_token_id(model, tokenizer):
    for source in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        if source is None:
            continue
        eos_token_id = getattr(source, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
    return tokenizer.eos_token_id


def format_chat_prompt(prompt: str, tokenizer, template_type: str) -> str:
    template_type = (template_type or "auto").lower()
    if template_type in {"none", "plain"}:
        return prompt
    if template_type == "llama2":
        return f"[INST]{prompt}[/INST]"
    if template_type in {"chatglm", "chatglm2"} and hasattr(tokenizer, "build_prompt"):
        return tokenizer.build_prompt(prompt)
    if template_type in {"xgen"}:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f" ### Human: {prompt}\n###"
    if template_type in {"internlm"}:
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except Exception:
            pass

    if template_type in {"vicuna", "longchat"}:
        return f"USER: {prompt}\nASSISTANT:"

    return prompt


def post_process_response(response: str, template_type: str) -> str:
    template_type = (template_type or "auto").lower()
    if template_type == "xgen":
        return response.strip().replace("Assistant:", "")
    if template_type == "internlm":
        return response.split("<eoa>")[0]
    return response


class TorchModel:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        *,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        device_map: str = "auto",
        compression_method: Optional[str] = None,
        triattention_stats_path: Optional[str] = None,
        triattention_budget: int = 2048,
        triattention_frequency_window: int = 65536,
        triattention_score_aggregation: str = "mean",
        triattention_divide_length: int = 128,
        triattention_disable_mlr: bool = False,
        triattention_disable_trig: bool = False,
    ) -> None:
        self.use_compass_loader = compression_method == "triattention"
        if self.use_compass_loader:
            from pathlib import Path
            from compass.src.TriAttention import apply_triattention_patch

            self.tokenizer = load_tokenizer_with_fallback(
                tokenizer_name_or_path or model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=_resolve_dtype(dtype) if torch.cuda.is_available() else _resolve_dtype(dtype),
                low_cpu_mem_usage=True,
                device_map=device_map,
                use_cache=True,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            self.model.eval()
            if not triattention_stats_path:
                raise ValueError("triattention_stats_path must be provided when compression_method='triattention'")
            apply_triattention_patch(
                self.model,
                stats_path=Path(triattention_stats_path).expanduser(),
                model_path=Path(model_name_or_path),
                kv_budget=int(triattention_budget),
                offset_max_length=int(triattention_frequency_window),
                score_aggregation=triattention_score_aggregation,
                pruning_seed=0,
                metadata_expectations={},
                normalize_scores=True,
                count_prompt_tokens=True,
                allow_prefill_compression=False,
                divide_length=int(triattention_divide_length),
                use_slack_trigger=True,
                per_head_pruning=True,
                per_layer_perhead_pruning=False,
                layer_perhead_aggregation="max",
                disable_mlr=bool(triattention_disable_mlr),
                disable_trig=bool(triattention_disable_trig),
            )
        else:
            self.tokenizer = load_tokenizer_with_fallback(
                tokenizer_name_or_path or model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            model_kwargs = {"trust_remote_code": trust_remote_code}
            if torch.cuda.is_available():
                model_kwargs["dtype"] = _resolve_dtype(dtype)
                model_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            self.model.eval()

        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            else:
                self.tokenizer.pad_token = fallback
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(fallback)
        self.tokenizer.padding_side = "left"
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eos_token_id = _resolve_eos_token_id(self.model, self.tokenizer)

    def generate(self, request: GenerationRequest, *, template_type: str = "auto") -> str:
        prompt = format_chat_prompt(request.prompt, self.tokenizer, template_type)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False, verbose=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        do_sample = request.temperature > 0
        generation_kwargs = dict(
            max_new_tokens=request.max_new_tokens,
            num_beams=1,
            do_sample=do_sample,
            temperature=max(request.temperature, 1e-5) if do_sample else None,
            top_p=request.top_p if do_sample else None,
            top_k=request.top_k if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        if request.dataset == "samsum":
            newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
            eos_ids = generation_kwargs["eos_token_id"]
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]
            if newline_token:
                eos_ids.append(newline_token[-1])
            generation_kwargs["eos_token_id"] = eos_ids
            generation_kwargs["min_length"] = prompt_len + 1
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)[0]
        text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        text = post_process_response(text, template_type)
        return _trim_stop_strings(text, request.stop)


class NanoVLLMModel:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        max_model_len: int,
        enable_cpu_offload: bool = False,
        num_gpu_blocks: int = 2,
        gpu_memory_utilization: float = 0.9,
        kvcache_block_size: int = 4096,
        enforce_eager: bool = True,
        sparse_policy: str = "FULL",
        sparse_stride: int = 8,
        sparse_threshold: float = 0.9,
        sparse_chunk_size: int = 16384,
        compass_top_p: Optional[float] = None,
        compass_lambda: Optional[float] = None,
        compass_theta: Optional[float] = None,
        blasst_fixed_lambda: Optional[float] = None,
    ) -> None:
        from nanovllm import LLM
        from nanovllm.config import SparsePolicyType

        policy = getattr(SparsePolicyType, sparse_policy.upper(), SparsePolicyType.FULL)
        kwargs = {
            "dtype": dtype,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
            "kvcache_block_size": kvcache_block_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "sparse_policy": policy,
            "sparse_stride": sparse_stride,
            "sparse_threshold": sparse_threshold,
            "sparse_chunk_size": sparse_chunk_size,
        }
        if enable_cpu_offload:
            kwargs["enable_cpu_offload"] = True
            kwargs["num_gpu_blocks"] = num_gpu_blocks
        if compass_top_p is not None:
            kwargs["compass_top_p"] = compass_top_p
        if compass_lambda is not None:
            kwargs["lambda_threshold"] = compass_lambda
        if compass_theta is not None:
            kwargs["compass_theta"] = compass_theta
        if blasst_fixed_lambda is not None:
            kwargs["blasst_fixed_lambda"] = blasst_fixed_lambda

        self.llm = LLM(model_name_or_path, **kwargs)
        self.tokenizer = self.llm.tokenizer
        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            else:
                self.tokenizer.pad_token = fallback
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(fallback)

    def generate(self, request: GenerationRequest, *, template_type: str = "auto") -> str:
        from nanovllm import SamplingParams

        prompt = format_chat_prompt(request.prompt, self.tokenizer, template_type)
        params = SamplingParams(
            temperature=max(request.temperature, 0.01),
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_new_tokens,
        )
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        text = outputs[0]["text"]
        text = post_process_response(text, template_type)
        return _trim_stop_strings(text, request.stop)
