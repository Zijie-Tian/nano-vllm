from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from upstream_utils import dataset_metric_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMPASS LongBench evaluator")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--e", action="store_true")
    return parser.parse_args()


def scorer_e(dataset: str, predictions: list[str], answers: list[list[str]], lengths: list[int], all_classes, dataset2metric):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        bucket = "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"
        scores[bucket].append(score)
    return {key: round(100 * np.mean(values), 2) if values else 0.0 for key, values in scores.items()}


def scorer(dataset: str, predictions: list[str], answers: list[list[str]], all_classes, dataset2metric):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def main() -> None:
    args = parse_args()
    dataset2metric = dataset_metric_map()
    output_root = Path(args.output_root)
    pred_dir = output_root / ("pred_e" if args.e else "pred") / args.model_name
    scores = {}
    for jsonl_file in sorted(pred_dir.glob("*.jsonl")):
        predictions, answers, lengths = [], [], []
        dataset = jsonl_file.stem
        all_classes = None
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                predictions.append(row["pred"])
                answers.append(row["answers"])
                all_classes = row["all_classes"]
                lengths.append(row.get("length", 0))
        if args.e:
            scores[dataset] = scorer_e(dataset, predictions, answers, lengths, all_classes, dataset2metric)
        else:
            scores[dataset] = scorer(dataset, predictions, answers, all_classes, dataset2metric)
    result_path = pred_dir / "result.json"
    result_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"Wrote evaluation to {result_path}")


if __name__ == "__main__":
    main()
