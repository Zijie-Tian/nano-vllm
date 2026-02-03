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
    # 4096
    # 8192
    # 16384
    # 32768
    # 65536
    # 131072
    # 262144   # 256K
    # 524288   # 512K
    786432   # 768K
    # 1048576  # 1M
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3

    case $MODEL_NAME in
        llama3.1-8b-chat)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="hf"
            ;;
        # NanoVLLM models (with CPU offload support)
        qwen3-0.6b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen3-0.6B"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            TOKENIZER_PATH="${MODEL_DIR}/Qwen3-0.6B"
            TOKENIZER_TYPE="hf"
            ;;
        qwen3-4b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen3-4B-Instruct-2507"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            ;;
        llama3.1-8b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="nanovllm"
            ;;
        # GLM-4-9B-Chat-1M (NanoVLLM backend)
        glm4-9b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            MODEL_TEMPLATE_TYPE="glm4"
            MODEL_FRAMEWORK="nanovllm"
            # GLM-4 uses HuggingFace tokenizer (not SentencePiece)
            TOKENIZER_PATH="${MODEL_DIR}/GLM-4-9B-Chat-1M"
            TOKENIZER_TYPE="hf"
            ;;
        # Qwen2.5-7B-Instruct-1M (NanoVLLM backend)
        qwen2.5-7b-1m-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct-1M"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            # Qwen2.5 uses HuggingFace tokenizer
            TOKENIZER_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct-1M"
            TOKENIZER_TYPE="hf"
            ;;
        # Qwen2.5-7B-Instruct (standard version, NanoVLLM backend)
        qwen2.5-7b-nanovllm)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen"
            MODEL_FRAMEWORK="nanovllm"
            # Qwen2.5 uses HuggingFace tokenizer
            TOKENIZER_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
            TOKENIZER_TYPE="hf"
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


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
