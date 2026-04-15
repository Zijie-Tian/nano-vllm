from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LONG_BENCH_ROOT = SCRIPT_DIR.parent
UPSTREAM_V1_ROOT = LONG_BENCH_ROOT / "upstream" / "LongBench"
UPSTREAM_CONFIG_DIR = UPSTREAM_V1_ROOT / "config"

LONG_BENCH_DATASETS = [
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

LONG_BENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
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


def ensure_upstream_exists() -> None:
    if not UPSTREAM_V1_ROOT.exists():
        raise FileNotFoundError(
            f"LongBench upstream assets not found at {UPSTREAM_V1_ROOT}. "
            "Did you initialize the eval/LongBench/upstream submodule?"
        )


def load_json_config(name: str) -> dict[str, Any]:
    ensure_upstream_exists()
    return json.loads((UPSTREAM_CONFIG_DIR / name).read_text(encoding="utf-8"))


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = None


def metrics_module() -> ModuleType:
    global _METRICS
    if _METRICS is None:
        ensure_upstream_exists()
        try:
            _METRICS = _load_module(UPSTREAM_V1_ROOT / "metrics.py", "_longbench_metrics")
        except ModuleNotFoundError:
            _METRICS = _load_module(SCRIPT_DIR / "fallback_metrics.py", "_longbench_fallback_metrics")
    return _METRICS


def dataset_metric_map() -> dict[str, Any]:
    metrics = metrics_module()
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
        "passage_retrieval_en": metrics.retrieval_score,
        "passage_count": metrics.count_score,
        "passage_retrieval_zh": metrics.retrieval_zh_score,
        "lcc": metrics.code_sim_score,
        "repobench-p": metrics.code_sim_score,
    }
