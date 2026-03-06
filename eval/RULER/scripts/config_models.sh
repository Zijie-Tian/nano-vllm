# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    # 4096     # 4K
    # 8192     # 8K
    # 16384    # 16K
    32768    # 32K
    # 65536
    # 131072
    # 262144   # 256K
    # 524288   # 512K
    # 786432   # 768K
    # 1048576  # 1M
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3

    # Reset variables for each call
    TOKENIZER_PATH=""
    TOKENIZER_TYPE=""
    NANOVLLM_CPU_OFFLOAD=""

    case $MODEL_NAME in
        llama3.1-8b-chat)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="hf"
            ;;
        # NanoVLLM models
        # NANOVLLM_CPU_OFFLOAD: true=use CPU offload (for large models), false=GPU only (for small models)
        qwen3-0.6b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen3-0.6B"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Qwen3-0.6B"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="false"  # Small model, GPU only (24GB sufficient for 32K)
            ;;
        qwen3-0.6b-xattn-nanovllm)
            # XAttention variant: use with --metric xattn for sparse attention testing
            MODEL_PATH="${MODEL_DIR}/Qwen3-0.6B"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Qwen3-0.6B"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="false"  # Small model, GPU only
            ;;
        qwen3-4b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen3-4B-Instruct-2507"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            NANOVLLM_CPU_OFFLOAD="true"  # Medium model, use offload for long context
            ;;
        llama3.1-8b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="nanovllm"
            NANOVLLM_CPU_OFFLOAD="true"  # Large model, use offload
            ;;
        llama3.1-8b-xattn-nanovllm)
            # XAttention variant: use with --metric xattn for sparse attention testing
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="true"  # Use offload for XAttention with long context
            ;;
        # GLM-4-9B-Chat-1M (NanoVLLM backend)
        glm4-9b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            MODEL_TEMPLATE_TYPE="glm4"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="true"  # Large model, use offload
            ;;
        glm4-9b-xattn-nanovllm)
            # XAttention variant: use with --metric xattn for sparse attention testing
            MODEL_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            MODEL_TEMPLATE_TYPE="glm4"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="true"  # Use offload for XAttention with long context
            ;;
        # Qwen2.5-7B-Instruct-1M (NanoVLLM backend)
        qwen2.5-7b-1m-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct-1M"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct-1M"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="true"  # Large model, use offload
            ;;
        # Qwen2.5-7B-Instruct (standard version, NanoVLLM backend)
        qwen2.5-7b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
            TOKENIZER_TYPE="hf"
            NANOVLLM_CPU_OFFLOAD="true"  # Large model, use offload
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi

    # Default NANOVLLM_CPU_OFFLOAD to true if not set (for backward compatibility)
    if [ -z "${NANOVLLM_CPU_OFFLOAD}" ]; then
        NANOVLLM_CPU_OFFLOAD="true"
    fi

    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT:$NANOVLLM_CPU_OFFLOAD"
}
