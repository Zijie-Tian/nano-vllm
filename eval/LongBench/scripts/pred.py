from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from model_wrappers import GenerationRequest, NanoVLLMModel, TorchModel, load_tokenizer_with_fallback
from upstream_utils import (
    LONG_BENCH_DATASETS,
    LONG_BENCH_E_DATASETS,
    NO_CHAT_WRAP_DATASETS,
    load_json_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMPASS LongBench predictor")
    parser.add_argument("--backend", choices=["torch", "nanovllm"], default="torch")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--model-name")
    parser.add_argument("--template-type", default="auto")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", help="Optional local LongBench data root (jsonl files or data.zip)")
    parser.add_argument("--datasets", help="Comma-separated dataset list")
    parser.add_argument("--e", action="store_true", help="Run LongBench-E")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--compression-method", choices=["triattention"])
    parser.add_argument("--triattention-stats-path")
    parser.add_argument("--triattention-budget", type=int, default=2048)
    parser.add_argument("--triattention-frequency-window", type=int, default=65536)
    parser.add_argument("--triattention-score-aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--triattention-divide-length", type=int, default=128)
    parser.add_argument("--triattention-disable-mlr", action="store_true")
    parser.add_argument("--triattention-disable-trig", action="store_true")
    parser.add_argument("--nanovllm-cpu-offload", action="store_true")
    parser.add_argument("--nanovllm-num-gpu-blocks", type=int, default=2)
    parser.add_argument("--nanovllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--nanovllm-block-size", type=int, default=4096)
    parser.add_argument("--nanovllm-enforce-eager", action="store_true")
    parser.add_argument("--nanovllm-sparse-policy", default="FULL")
    parser.add_argument("--nanovllm-sparse-stride", type=int, default=8)
    parser.add_argument("--nanovllm-sparse-threshold", type=float, default=0.9)
    parser.add_argument("--nanovllm-sparse-chunk-size", type=int, default=16384)
    parser.add_argument("--nanovllm-compass-top-p", type=float)
    parser.add_argument("--nanovllm-compass-lambda", type=float)
    parser.add_argument("--nanovllm-compass-theta", type=float)
    parser.add_argument("--blasst-lambda", type=float)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_datasets(raw: str | None, e_mode: bool) -> list[str]:
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(LONG_BENCH_E_DATASETS if e_mode else LONG_BENCH_DATASETS)


def infer_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return Path(args.model_path.rstrip("/")).name


def maybe_truncate_prompt(prompt: str, tokenizer, max_model_len: int) -> str:
    tokens = tokenizer(prompt, truncation=False, return_tensors="pt", verbose=False).input_ids[0]
    if len(tokens) <= max_model_len:
        return prompt
    half = max_model_len // 2
    left = tokenizer.decode(tokens[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokens[-half:], skip_special_tokens=True)
    return left + right


def resolve_max_model_len(args: argparse.Namespace, tokenizer, model_name: str) -> int:
    if args.max_model_len:
        return args.max_model_len
    upstream = load_json_config("model2maxlen.json")
    if model_name in upstream:
        return int(upstream[model_name])
    candidate = getattr(tokenizer, "model_max_length", None)
    if isinstance(candidate, int) and 0 < candidate < 10**7:
        return int(candidate)
    return 32768


def build_backend(args: argparse.Namespace):
    trust_remote_code = not args.no_trust_remote_code
    if args.backend == "torch":
        return TorchModel(
            args.model_path,
            tokenizer_name_or_path=args.tokenizer_path,
            dtype=args.dtype,
            trust_remote_code=trust_remote_code,
            compression_method=args.compression_method,
            triattention_stats_path=args.triattention_stats_path,
            triattention_budget=args.triattention_budget,
            triattention_frequency_window=args.triattention_frequency_window,
            triattention_score_aggregation=args.triattention_score_aggregation,
            triattention_divide_length=args.triattention_divide_length,
            triattention_disable_mlr=args.triattention_disable_mlr,
            triattention_disable_trig=args.triattention_disable_trig,
        )

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = load_tokenizer_with_fallback(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    max_model_len = resolve_max_model_len(args, tokenizer, infer_model_name(args))
    return NanoVLLMModel(
        args.model_path,
        dtype=args.dtype,
        max_model_len=max_model_len,
        enable_cpu_offload=args.nanovllm_cpu_offload,
        num_gpu_blocks=args.nanovllm_num_gpu_blocks,
        gpu_memory_utilization=args.nanovllm_gpu_memory_utilization,
        kvcache_block_size=args.nanovllm_block_size,
        enforce_eager=args.nanovllm_enforce_eager,
        sparse_policy=args.nanovllm_sparse_policy,
        sparse_stride=args.nanovllm_sparse_stride,
        sparse_threshold=args.nanovllm_sparse_threshold,
        sparse_chunk_size=args.nanovllm_sparse_chunk_size,
        compass_top_p=args.nanovllm_compass_top_p,
        compass_lambda=args.nanovllm_compass_lambda,
        compass_theta=args.nanovllm_compass_theta,
        blasst_fixed_lambda=args.blasst_lambda,
    )


def iter_data_roots(cli_data_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for raw in (
        cli_data_root,
        os.environ.get("LONG_BENCH_DATA_ROOT"),
        str(Path.home() / "data" / "LongBench"),
        str(Path.home() / "data" / "longbench"),
    ):
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    return candidates


def load_local_subset(data_root: Path, subset: str, num_samples: int | None) -> list[dict] | None:
    jsonl_candidates = (
        data_root / "data" / f"{subset}.jsonl",
        data_root / f"{subset}.jsonl",
    )
    for jsonl_path in jsonl_candidates:
        if jsonl_path.exists():
            rows = []
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for idx, raw_line in enumerate(fh):
                    if num_samples is not None and idx >= num_samples:
                        break
                    rows.append(json.loads(raw_line))
            return rows

    zip_candidates = (
        data_root / "data.zip",
        data_root / "LongBench" / "data.zip",
    )
    for zip_path in zip_candidates:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                member = f"data/{subset}.jsonl"
                if member not in zf.namelist():
                    continue
                rows = []
                with zf.open(member) as fh:
                    for idx, raw_line in enumerate(fh):
                        if num_samples is not None and idx >= num_samples:
                            break
                        rows.append(json.loads(raw_line.decode("utf-8")))
                return rows

    return None


def iterate_samples(dataset_name: str, e_mode: bool, num_samples: int | None, data_root: str | None) -> Iterable[dict]:
    subset = f"{dataset_name}_e" if e_mode else dataset_name
    for candidate_root in iter_data_roots(data_root):
        rows = load_local_subset(candidate_root, subset, num_samples)
        if rows is not None:
            return rows

    try:
        data = load_dataset("THUDM/LongBench", subset, split="test")
        if num_samples is not None:
            data = data.select(range(min(num_samples, len(data))))
        return data
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        zip_path = hf_hub_download(
            repo_id="THUDM/LongBench",
            repo_type="dataset",
            filename="data.zip",
        )
        rows = []
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"data/{subset}.jsonl") as fh:
                for idx, raw_line in enumerate(fh):
                    if num_samples is not None and idx >= num_samples:
                        break
                    rows.append(json.loads(raw_line.decode("utf-8")))
        return rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model_name = infer_model_name(args)
    backend = build_backend(args)
    max_model_len = resolve_max_model_len(args, backend.tokenizer, model_name)
    datasets = parse_datasets(args.datasets, args.e)
    dataset2prompt = load_json_config("dataset2prompt.json")
    dataset2maxlen = load_json_config("dataset2maxlen.json")

    output_root = Path(args.output_root)
    pred_parent = output_root / ("pred_e" if args.e else "pred") / model_name
    pred_parent.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        output_file = pred_parent / f"{dataset_name}.jsonl"
        if output_file.exists():
            output_file.unlink()
        samples = list(iterate_samples(dataset_name, args.e, args.num_samples, args.data_root))
        prompt_template = dataset2prompt[dataset_name]
        max_new_tokens = int(dataset2maxlen[dataset_name])

        for sample in tqdm(samples, desc=f"{dataset_name}", leave=False):
            prompt = prompt_template.format(**sample)
            prompt = maybe_truncate_prompt(prompt, backend.tokenizer, max_model_len)
            if dataset_name in NO_CHAT_WRAP_DATASETS:
                prompt_for_model = prompt
                template_type = "plain"
            else:
                prompt_for_model = prompt
                template_type = args.template_type
            pred = backend.generate(
                GenerationRequest(
                    prompt=prompt_for_model,
                    max_new_tokens=max_new_tokens,
                    dataset=dataset_name,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                ),
                template_type=template_type,
            )
            with output_file.open("a", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pred": pred,
                        "answers": sample["answers"],
                        "all_classes": sample["all_classes"],
                        "length": sample["length"],
                    },
                    fh,
                    ensure_ascii=False,
                )
                fh.write("\n")

    print(f"Wrote predictions to {pred_parent}")


if __name__ == "__main__":
    main()
