#!/bin/bash

NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Full LongBench task list
LONG_BENCH_ALL=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "multifieldqa_zh"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "dureader"
    "gov_report"
    "qmsum"
    "multi_news"
    "vcsum"
    "trec"
    "triviaqa"
    "samsum"
    "lsht"
    "passage_count"
    "passage_retrieval_en"
    "passage_retrieval_zh"
    "lcc"
    "repobench-p"
)

# TriAttention subset requested by user:
# NarrQA Qasp MFQA HpQA 2Wik Musi GovR QMSu MNew TREC TriQA SSum PaRe PaCn LCC ReBe
LONG_BENCH_TRIATTENTION=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "gov_report"
    "qmsum"
    "multi_news"
    "trec"
    "triviaqa"
    "samsum"
    "passage_retrieval_en"
    "passage_count"
    "lcc"
    "repobench-p"
)

TASKS_SELECT() {
    local task_set="$1"

    case "$task_set" in
        all)
            echo "${LONG_BENCH_ALL[*]}"
            ;;
        triattention)
            echo "${LONG_BENCH_TRIATTENTION[*]}"
            ;;
        *)
            # Allow direct single-dataset or comma-separated overrides outside the named presets
            echo "$task_set"
            ;;
    esac
}
