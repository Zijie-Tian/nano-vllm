from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/zijie/Code/triattention')
sys.path.insert(0, '/home/zijie/Code/COMPASS/eval/LongBench/scripts')

from triattention.methods.triattention import apply_triattention_patch
from upstream_utils import load_json_config, NO_CHAT_WRAP_DATASETS
from model_wrappers import format_chat_prompt, post_process_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run standard TriAttention on LongBench samples for alignment checks.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--stats-path', required=True)
    parser.add_argument('--data-root', default=str(Path.home() / 'data' / 'LongBench'))
    parser.add_argument('--datasets', required=True, help='Comma-separated datasets')
    parser.add_argument('--num-samples', type=int, default=2)
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument('--output-root', default='/home/zijie/Code/COMPASS/eval/LongBench/benchmark_root')
    parser.add_argument('--model-name', default='Qwen3-8B-standard-triattention')
    return parser.parse_args()


def maybe_truncate_prompt(prompt: str, tokenizer, max_model_len: int) -> str:
    tokens = tokenizer(prompt, truncation=False, return_tensors='pt', verbose=False).input_ids[0]
    if len(tokens) <= max_model_len:
        return prompt
    half = max_model_len // 2
    left = tokenizer.decode(tokens[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokens[-half:], skip_special_tokens=True)
    return left + right


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(',') if item.strip()]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    output_root = Path(args.output_root)
    pred_dir = output_root / 'pred' / args.model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    with zipfile.ZipFile(data_root / 'data.zip') as zf:
        dataset_rows = {}
        for dataset in datasets:
            rows = []
            with zf.open(f'data/{dataset}.jsonl') as fh:
                for idx, line in enumerate(fh):
                    if idx >= args.num_samples:
                        break
                    rows.append(json.loads(line.decode('utf-8')))
            dataset_rows[dataset] = rows

    dataset2prompt = load_json_config('dataset2prompt.json')
    dataset2maxlen = load_json_config('dataset2maxlen.json')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.unk_token
        if fallback is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = fallback
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(fallback)
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        use_cache=True,
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
    model.eval()
    apply_triattention_patch(
        model,
        stats_path=Path(args.stats_path),
        model_path=Path(args.model_path),
        kv_budget=2048,
        offset_max_length=65536,
        score_aggregation='mean',
        pruning_seed=0,
        metadata_expectations={},
        normalize_scores=True,
        count_prompt_tokens=True,
        allow_prefill_compression=False,
        divide_length=128,
        use_slack_trigger=True,
        per_head_pruning=True,
        per_layer_perhead_pruning=False,
        layer_perhead_aggregation='max',
        disable_mlr=False,
        disable_trig=False,
    )

    for dataset in datasets:
        out_file = pred_dir / f'{dataset}.jsonl'
        if out_file.exists():
            out_file.unlink()
        prompt_template = dataset2prompt[dataset]
        max_new_tokens = int(dataset2maxlen[dataset])
        for sample in dataset_rows[dataset]:
            prompt = prompt_template.format(**sample)
            prompt = maybe_truncate_prompt(prompt, tokenizer, args.max_model_len)
            template_type = 'plain' if dataset in NO_CHAT_WRAP_DATASETS else 'auto'
            if template_type != 'plain':
                prompt = format_chat_prompt(prompt, tokenizer, template_type)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=False, verbose=False)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs['input_ids'].shape[-1]
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]
            text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            text = post_process_response(text, template_type)
            with out_file.open('a', encoding='utf-8') as fh:
                json.dump({'pred': text, 'answers': sample['answers'], 'all_classes': sample['all_classes'], 'length': sample['length']}, fh, ensure_ascii=False)
                fh.write('\n')

    print(pred_dir)


if __name__ == '__main__':
    main()
