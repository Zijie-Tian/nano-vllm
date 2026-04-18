"""
Independent LongBench runner using upstream prompts/metrics and local data.

This script is intentionally separate from eval/LongBench/scripts/run.sh and
pred.py so we can compare results against a more upstream-like path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UPSTREAM_ROOT = REPO_ROOT / "eval" / "LongBench" / "upstream" / "LongBench"
UPSTREAM_CONFIG = UPSTREAM_ROOT / "config"
FALLBACK_METRICS = REPO_ROOT / "eval" / "LongBench" / "scripts" / "fallback_metrics.py"

ALL_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

NO_CHAT_WRAP_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}


# ============================================================
# Utility Functions
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent LongBench runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit model device like cuda:0 or cpu. Defaults to device_map=auto when unset.",
    )
    parser.add_argument("--data-root", default=str(Path.home() / "data" / "LongBench"))
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "eval" / "LongBench" / "benchmark_root" / "standard_runner"),
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_upstream_exists() -> None:
    if not UPSTREAM_CONFIG.exists():
        raise FileNotFoundError(
            f"LongBench upstream config not found at {UPSTREAM_CONFIG}. "
            "Did you initialize the eval/LongBench/upstream submodule?"
        )


def load_metrics_module() -> ModuleType:
    upstream_metrics = UPSTREAM_ROOT / "metrics.py"
    candidates = [upstream_metrics, FALLBACK_METRICS] if upstream_metrics.exists() else [FALLBACK_METRICS]
    last_error = None
    for target in candidates:
        try:
            spec = importlib.util.spec_from_file_location("_longbench_metrics_independent", target)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load metrics module from {target}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except ModuleNotFoundError as exc:
            last_error = exc
    raise last_error if last_error is not None else RuntimeError("Failed to load LongBench metrics module")


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    attempts = [
        {"trust_remote_code": True, "use_fast": False},
        {"trust_remote_code": True},
        {"trust_remote_code": True, "use_fast": True},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            return tokenizer
        last_error = TypeError(
            "AutoTokenizer.from_pretrained returned "
            f"{type(tokenizer).__name__} for {model_path} with kwargs={kwargs}"
        )
    raise RuntimeError(f"Failed to load a valid tokenizer from {model_path}") from last_error


def dataset_metric_map():
    metrics = load_metrics_module()
    return {
        "narrativeqa": metrics.qa_f1_score,
        "qasper": metrics.qa_f1_score,
        "multifieldqa_en": metrics.qa_f1_score,
        "multifieldqa_zh": metrics.qa_f1_zh_score,
        "hotpotqa": metrics.qa_f1_score,
        "2wikimqa": metrics.qa_f1_score,
        "musique": metrics.qa_f1_score,
        "dureader": metrics.rouge_zh_score,
        "gov_report": metrics.rouge_score,
        "qmsum": metrics.rouge_score,
        "multi_news": metrics.rouge_score,
        "vcsum": metrics.rouge_zh_score,
        "trec": metrics.classification_score,
        "triviaqa": metrics.qa_f1_score,
        "samsum": metrics.rouge_score,
        "lsht": metrics.classification_score,
        "passage_count": metrics.count_score,
        "passage_retrieval_en": metrics.retrieval_score,
        "passage_retrieval_zh": metrics.retrieval_zh_score,
        "lcc": metrics.code_sim_score,
        "repobench-p": metrics.code_sim_score,
    }


def select_datasets(raw: str) -> list[str]:
    if raw == "all":
        return list(ALL_DATASETS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def iter_local_rows(data_root: Path, dataset: str, num_samples: int) -> list[dict]:
    jsonl_candidates = [
        data_root / "data" / f"{dataset}.jsonl",
        data_root / f"{dataset}.jsonl",
    ]
    for candidate in jsonl_candidates:
        if candidate.exists():
            rows = []
            with candidate.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx >= num_samples:
                        break
                    rows.append(json.loads(line))
            return rows

    zip_candidates = [
        data_root / "data.zip",
        data_root / "LongBench" / "data.zip",
    ]
    for candidate in zip_candidates:
        if candidate.exists():
            with zipfile.ZipFile(candidate) as zf:
                member = f"data/{dataset}.jsonl"
                if member not in zf.namelist():
                    continue
                rows = []
                with zf.open(member) as fh:
                    for idx, line in enumerate(fh):
                        if idx >= num_samples:
                            break
                        rows.append(json.loads(line.decode("utf-8")))
                return rows

    raise FileNotFoundError(f"Could not find local LongBench data for {dataset} under {data_root}")


def build_chat(prompt: str, tokenizer, model_name: str) -> str:
    lower = model_name.lower()
    if "chatglm3" in lower:
        return tokenizer.build_chat_input(prompt)
    if "chatglm" in lower and hasattr(tokenizer, "build_prompt"):
        return tokenizer.build_prompt(prompt)
    if "llama2" in lower:
        return f"[INST]{prompt}[/INST]"
    if "xgen" in lower:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f" ### Human: {prompt}\n###"
    if "internlm" in lower:
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
    return prompt


def post_process(response: str, model_name: str) -> str:
    lower = model_name.lower()
    if "xgen" in lower:
        return response.strip().replace("Assistant:", "")
    if "internlm" in lower:
        return response.split("<eoa>")[0]
    return response


def detect_max_model_len(model_path: Path, tokenizer, override: int | None) -> int:
    if override:
        return override
    config_path = model_path / "config.json"
    if config_path.exists():
        config = load_json(config_path)
        for key in ("max_position_embeddings", "seq_length", "model_max_length"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value
    candidate = getattr(tokenizer, "model_max_length", None)
    if isinstance(candidate, int) and 0 < candidate < 10**7:
        return candidate
    return 32768


def truncate_prompt(prompt: str, tokenizer, max_model_len: int) -> str:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt", verbose=False).input_ids[0]
    if len(tokenized) <= max_model_len:
        return prompt
    half = max_model_len // 2
    left = tokenizer.decode(tokenized[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    return left + right


def load_model_and_tokenizer(model_path: Path, dtype: str, device: str | None):
    torch_dtype = getattr(torch, dtype)
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.unk_token
        if fallback is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = fallback
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(fallback)
    tokenizer.padding_side = "left"
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if device:
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    resolved_device = next(model.parameters()).device
    return model, tokenizer, resolved_device


def generate_prediction(model, tokenizer, device, prompt: str, dataset: str, max_gen: int, model_name: str) -> str:
    prompt_text = prompt if dataset in NO_CHAT_WRAP_DATASETS else build_chat(prompt, tokenizer, model_name)
    inputs = tokenizer(prompt_text, truncation=False, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    context_length = inputs["input_ids"].shape[-1]

    if dataset == "samsum":
        newline_token = tokenizer.encode("\n", add_special_tokens=False)
        eos_ids = [tokenizer.eos_token_id]
        if newline_token:
            eos_ids.append(newline_token[-1])
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length + 1,
            eos_token_id=eos_ids,
        )[0]
    else:
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]

    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return post_process(pred, model_name)


def score_dataset(dataset: str, predictions: list[str], answers: list[list[str]], lengths: list[int], classes, metric_map) -> float:
    total = 0.0
    for pred, gts in zip(predictions, answers):
        score = 0.0
        normalized_pred = pred
        if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
            normalized_pred = pred.lstrip("\n").split("\n")[0]
        for gt in gts:
            score = max(score, metric_map[dataset](normalized_pred, gt, all_classes=classes))
        total += score
    return round(100 * total / len(predictions), 2) if predictions else 0.0


# ============================================================
# Main Test Script
# ============================================================

args = parse_args()
seed_everything(args.seed)

model_path = Path(args.model_path).expanduser().resolve()
model_name = args.model_name or model_path.name
data_root = Path(args.data_root).expanduser().resolve()
datasets = select_datasets(args.datasets)
output_root = Path(args.output_root).expanduser().resolve()
pred_dir = output_root / "pred" / model_name
pred_dir.mkdir(parents=True, exist_ok=True)

ensure_upstream_exists()
dataset2prompt = load_json(UPSTREAM_CONFIG / "dataset2prompt.json")
dataset2maxlen = load_json(UPSTREAM_CONFIG / "dataset2maxlen.json")
metric_map = dataset_metric_map()

model, tokenizer, device = load_model_and_tokenizer(model_path, args.dtype, args.device)
max_model_len = detect_max_model_len(model_path, tokenizer, args.max_model_len)

scores = {}

for dataset in datasets:
    rows = iter_local_rows(data_root, dataset, args.num_samples)
    prompt_format = dataset2prompt[dataset]
    max_gen = int(dataset2maxlen[dataset])
    out_path = pred_dir / f"{dataset}.jsonl"
    if out_path.exists():
        out_path.unlink()

    predictions, answers, lengths = [], [], []
    all_classes = None

    for row in rows:
        prompt = prompt_format.format(**row)
        prompt = truncate_prompt(prompt, tokenizer, max_model_len)
        pred = generate_prediction(model, tokenizer, device, prompt, dataset, max_gen, model_name)
        predictions.append(pred)
        answers.append(row["answers"])
        all_classes = row["all_classes"]
        lengths.append(row.get("length", 0))
        with out_path.open("a", encoding="utf-8") as fh:
            json.dump(
                {
                    "pred": pred,
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row.get("length", 0),
                },
                fh,
                ensure_ascii=False,
            )
            fh.write("\n")

    scores[dataset] = score_dataset(dataset, predictions, answers, lengths, all_classes, metric_map)

result_path = pred_dir / "result.json"
result_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(scores, ensure_ascii=False, indent=2))
print(f"{Path(__file__).name}: PASSED")
