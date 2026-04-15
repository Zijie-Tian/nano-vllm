from __future__ import annotations

import re
import string
from collections import Counter
from difflib import SequenceMatcher


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［\］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in all_punctuation)
    return "".join(text.split())


def _tokenize_zh(text: str) -> list[str]:
    return [normalize_zh_answer(ch) for ch in text if normalize_zh_answer(ch)]


def f1_score(prediction_tokens, ground_truth_tokens, **kwargs):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    return f1_score(_tokenize_zh(prediction), _tokenize_zh(ground_truth))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if str(number) == str(ground_truth))
    return right / len(numbers)


def retrieval_score(prediction, ground_truth, **kwargs):
    match = re.findall(r"Paragraph (\d+)", ground_truth)
    target = match[0] if match else None
    numbers = re.findall(r"\d+", prediction)
    if not numbers or target is None:
        return 0.0
    right = sum(1 for number in numbers if number == target)
    return right / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    match = re.findall(r"段落(\d+)", ground_truth)
    target = match[0] if match else None
    numbers = re.findall(r"\d+", prediction)
    if not numbers or target is None:
        return 0.0
    right = sum(1 for number in numbers if number == target)
    return right / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if '`' not in line and '#' not in line and '//' not in line:
            candidate = line
            break
    return SequenceMatcher(None, candidate, ground_truth).ratio()


def classification_score(prediction, ground_truth, **kwargs):
    classes = kwargs.get("all_classes") or []
    matches = [class_name for class_name in classes if class_name in prediction]
    matches = [m for m in matches if not (m in ground_truth and m != ground_truth)]
    if ground_truth in matches:
        return 1.0 / len(matches)
    return 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_score(prediction, ground_truth, **kwargs):
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gt_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def rouge_zh_score(prediction, ground_truth, **kwargs):
    return rouge_score(" ".join(_tokenize_zh(prediction)), " ".join(_tokenize_zh(ground_truth)))
