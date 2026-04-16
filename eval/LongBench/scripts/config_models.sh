#!/bin/bash

# Default generation settings
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-32}"
DTYPE="${DTYPE:-bfloat16}"

MODEL_SELECT() {
    local model_ref="$1"
    local model_dir="$2"

    local model_path=""
    local model_name=""
    local model_backend="torch"
    local tokenizer_path=""
    local template_type="auto"
    local dtype="${DTYPE}"
    local max_model_len=""
    local compression_method=""
    local triattention_stats_path=""
    local triattention_budget=""

    case "$model_ref" in
        tiny-gpt2)
            model_path="sshleifer/tiny-gpt2"
            model_name="tiny-gpt2"
            template_type="plain"
            dtype="float32"
            max_model_len="1024"
            ;;
        llama3.1-8b-instruct)
            model_path="${model_dir}/Llama-3.1-8B-Instruct"
            model_name="Llama-3.1-8B-Instruct"
            template_type="auto"
            max_model_len="16384"
            ;;
        qwen3-8b)
            model_path="${model_dir}/Qwen3-8B"
            model_name="Qwen3-8B"
            template_type="auto"
            max_model_len="16384"
            ;;
        qwen3-8b-triattention)
            model_path="${model_dir}/Qwen3-8B"
            model_name="Qwen3-8B"
            template_type="auto"
            max_model_len="16384"
            compression_method="triattention"
            triattention_stats_path="${TRIATTENTION_STATS_PATH:-}"
            triattention_budget="${TRIATTENTION_BUDGET:-2048}"
            ;;
        llama3.1-8b-instruct-triattention)
            model_path="${model_dir}/Llama-3.1-8B-Instruct"
            model_name="Llama-3.1-8B-Instruct"
            template_type="auto"
            max_model_len="16384"
            compression_method="triattention"
            triattention_stats_path="${TRIATTENTION_STATS_PATH:-}"
            triattention_budget="${TRIATTENTION_BUDGET:-2048}"
            ;;
        llama3.1-nemotron-8b-ultralong-1m-instruct)
            model_path="${model_dir}/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct"
            model_name="Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct"
            template_type="auto"
            ;;
        *)
            model_path="$model_ref"
            model_name="$(basename "$model_ref")"
            ;;
    esac

    echo "${model_path}:${model_name}:${model_backend}:${tokenizer_path}:${template_type}:${dtype}:${max_model_len}:${compression_method}:${triattention_stats_path}:${triattention_budget}"
}
