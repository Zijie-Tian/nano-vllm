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

HAS_DATASET() {
    local candidate="$1"
    local dataset
    for dataset in "${LONG_BENCH_ALL[@]}"; do
        if [[ "$candidate" == "$dataset" ]]; then
            return 0
        fi
    done
    return 1
}

TASKS_SELECT() {
    local task_set="$1"
    local raw_item
    local item
    local resolved=()

    case "$task_set" in
        all)
            echo "${LONG_BENCH_ALL[*]}"
            ;;
        triattention)
            echo "${LONG_BENCH_TRIATTENTION[*]}"
            ;;
        *)
            IFS=',' read -ra raw_items <<< "$task_set"
            for raw_item in "${raw_items[@]}"; do
                item="$(echo "$raw_item" | xargs)"
                if [[ -z "$item" ]]; then
                    continue
                fi
                if ! HAS_DATASET "$item"; then
                    return 1
                fi
                resolved+=("$item")
            done
            if [[ ${#resolved[@]} -eq 0 ]]; then
                return 1
            fi
            echo "${resolved[*]}"
            ;;
    esac
}
