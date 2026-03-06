#!/bin/bash
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

# Usage: bash run.sh MODEL_NAME BENCHMARK_NAME [OPTIONS]


# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="${MODEL_DIR:-/home/zijie/models}" # the path that contains individual model folders from Huggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

# Set PYTHONPATH for COMPASS and 3rdparty modules
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/nanovllm:${PYTHONPATH}"

# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT NANOVLLM_CPU_OFFLOAD <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}
# NanoVLLM CPU offload setting (true/false, default true for backward compatibility)
export NANOVLLM_CPU_OFFLOAD=${NANOVLLM_CPU_OFFLOAD:-true}


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Parse additional arguments with defaults
METIRC=${METRIC:-"--metric xattn"} # Default: xattn
PRINT_DETAIL=${PRINT_DETAIL:-""}
STRIDE=${STRIDE:-""}
THRESHOLD=${THRESHOLD:-""}
AVGPOOL_TOPK=${AVGPOOL_TOPK:-""}
AVGPOOL_TOPP=${AVGPOOL_TOPP:-""}
TASK_OVERRIDE=""

shift 2 # Remove MODEL_NAME and BENCHMARK
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD="--threshold $2"
            shift 2
            ;;
        --metric)
            METRIC="--metric $2"
            shift 2
            ;;
        --print_detail)
            PRINT_DETAIL="--print_detail"
            shift
            ;;
        --stride)
            STRIDE="--stride $2"
            shift 2
            ;;
        --avgpool_topk)
            AVGPOOL_TOPK="--avgpool_topk $2"
            shift 2
            ;;
        --avgpool_topp)
            AVGPOOL_TOPP="--avgpool_topp $2"
            shift 2
            ;;
        --task)
            TASK_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Override TASKS if --task is specified
if [ -n "${TASK_OVERRIDE}" ]; then
    IFS=',' read -ra TASKS <<< "${TASK_OVERRIDE}"
    echo "Task override: ${TASKS[*]}"
fi

# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        &

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs

fi


# NanoVLLM parallel execution settings
# GPU configuration for parallel execution
GPU_LIST=${GPU_LIST:-"2,3,4,5"}  # Comma-separated GPU IDs to use (4 GPUs for parallel testing)
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    # Set max_model_len for nanovllm to 1M (covers all test lengths)
    export NANOVLLM_MAX_MODEL_LEN=1048576

    SETTINGS_INFO=""
    if [[ -n ${METRIC} ]]; then SETTINGS_INFO+="${METRIC#--metric }_"; fi
    # For avgpool: use topp or topk instead of stride
    # For xattn with nanovllm: use stride for folder naming
    METRIC_NAME="${METRIC#--metric }"
    if [[ "${METRIC_NAME}" == "avgpool" ]]; then
        # top-p takes priority over top-k for folder naming
        if [[ -n ${AVGPOOL_TOPP} ]]; then
            SETTINGS_INFO+="topp_${AVGPOOL_TOPP##* }_"
        elif [[ -n ${AVGPOOL_TOPK} ]]; then
            SETTINGS_INFO+="topk_${AVGPOOL_TOPK##* }_"
        fi
    elif [[ "${METRIC_NAME}" == "blasst" ]]; then
        # For BLASST, include lambda in folder name
        BLASST_LAMBDA_VAL="${BLASST_LAMBDA:-0.5}"
        SETTINGS_INFO+="lambda${BLASST_LAMBDA_VAL}_"
    else
        # For xattn (nanovllm or other backends), include stride in folder name
        if [[ -n ${STRIDE} ]]; then
            SETTINGS_INFO+="stride${STRIDE##* }_"
            # Export for nanovllm backend
            export NANOVLLM_XATTN_STRIDE="${STRIDE##* }"
        fi
    fi
    if [[ -n ${THRESHOLD} && -z ${PRECISE_THRESHOLD} ]]; then SETTINGS_INFO+="thresh_${THRESHOLD#--threshold }_"; fi

    RESULTS_DIR="${ROOT_DIR}/${SETTINGS_INFO}${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    # Prepare data for all tasks in parallel
    echo "Preparing data for ${#TASKS[@]} tasks in parallel..."
    declare -a DATA_PIDS
    declare -a DATA_TASKS

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB} &
        DATA_PIDS+=($!)
        DATA_TASKS+=("$TASK")
    done

    # Wait for all data preparation processes and check for failures
    DATA_FAILED=0
    for i in "${!DATA_PIDS[@]}"; do
        pid=${DATA_PIDS[$i]}
        task=${DATA_TASKS[$i]}
        if ! wait $pid; then
            echo "ERROR: Data preparation failed for task: $task (PID: $pid)"
            DATA_FAILED=1
        fi
    done

    if [ $DATA_FAILED -eq 1 ]; then
        echo "ERROR: One or more data preparation tasks failed. Exiting."
        exit 1
    fi
    echo "All data preparation tasks completed successfully."

    # ============================================================
    # DEBUG: Skip model inference, only test data generation
    # ============================================================
    # echo "DEBUG MODE: Skipping model inference and evaluation"
    # continue
    # ============================================================

    # NanoVLLM: parallel execution with GPU-locked scheduling
    if [ "$MODEL_FRAMEWORK" == "nanovllm" ]; then
        echo "NanoVLLM detected: using GPU-locked scheduling with ${NUM_GPUS} GPUs (${GPU_LIST})"
        start_time=$(date +%s)

        TASK_INDEX=0
        # Track PID for each GPU slot: GPU_PIDS[gpu_idx]=pid (0 means free)
        declare -a GPU_PIDS
        for ((i=0; i<NUM_GPUS; i++)); do
            GPU_PIDS[$i]=0
        done

        # Function to find a free GPU slot, returns -1 if none available
        find_free_gpu() {
            for ((i=0; i<NUM_GPUS; i++)); do
                local pid=${GPU_PIDS[$i]}
                if [ "$pid" -eq 0 ]; then
                    echo $i
                    return
                fi
                # Check if process is still running
                if ! kill -0 $pid 2>/dev/null; then
                    GPU_PIDS[$i]=0
                    echo $i
                    return
                fi
            done
            echo -1
        }

        # Function to count active GPUs
        count_active() {
            local count=0
            for ((i=0; i<NUM_GPUS; i++)); do
                local pid=${GPU_PIDS[$i]}
                if [ "$pid" -ne 0 ] && kill -0 $pid 2>/dev/null; then
                    ((count++))
                fi
            done
            echo $count
        }

        # Main loop: schedule tasks to free GPUs
        while [ $TASK_INDEX -lt ${#TASKS[@]} ] || [ $(count_active) -gt 0 ]; do
            # Try to launch tasks on free GPUs
            while [ $TASK_INDEX -lt ${#TASKS[@]} ]; do
                FREE_GPU=$(find_free_gpu)
                if [ "$FREE_GPU" -eq -1 ]; then
                    break  # No free GPU, wait
                fi

                TASK=${TASKS[$TASK_INDEX]}
                GPU_ID=${GPU_ARRAY[$FREE_GPU]}
                echo "  [${TASK_INDEX}/${#TASKS[@]}] Task ${TASK} -> GPU ${GPU_ID} (slot ${FREE_GPU})"

                CUDA_VISIBLE_DEVICES=${GPU_ID} python pred/call_api.py \
                    --data_dir ${DATA_DIR} \
                    --save_dir ${PRED_DIR} \
                    --benchmark ${BENCHMARK} \
                    --task ${TASK} \
                    --server_type ${MODEL_FRAMEWORK} \
                    --model_name_or_path ${MODEL_PATH} \
                    --temperature ${TEMPERATURE} \
                    --top_k ${TOP_K} \
                    --top_p ${TOP_P} \
                    --batch_size ${BATCH_SIZE} \
                    --num_samples ${NUM_SAMPLES} \
                    ${STOP_WORDS} \
                    ${METRIC} \
                    ${THRESHOLD} \
                    ${STRIDE} \
                    ${AVGPOOL_TOPK} \
                    ${AVGPOOL_TOPP} \
                    ${PRINT_DETAIL} &

                GPU_PIDS[$FREE_GPU]=$!
                TASK_INDEX=$((TASK_INDEX + 1))

                # Small delay to allow GPU initialization
                sleep 2
            done

            # Wait a bit before checking again
            if [ $(count_active) -gt 0 ]; then
                sleep 5
            fi
        done

        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))

    # Other backends: sequential execution
    else
        for TASK in "${TASKS[@]}"; do
            start_time=$(date +%s)
            python pred/call_api.py \
                --data_dir ${DATA_DIR} \
                --save_dir ${PRED_DIR} \
                --benchmark ${BENCHMARK} \
                --task ${TASK} \
                --server_type ${MODEL_FRAMEWORK} \
                --model_name_or_path ${MODEL_PATH} \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --top_p ${TOP_P} \
                --batch_size ${BATCH_SIZE} \
                --num_samples ${NUM_SAMPLES} \
                ${STOP_WORDS} \
                ${METRIC} \
                ${THRESHOLD} \
                ${STRIDE} \
                ${AVGPOOL_TOPK} \
                ${AVGPOOL_TOPP} \
                ${PRINT_DETAIL}
            end_time=$(date +%s)
            time_diff=$((end_time - start_time))
            total_time=$((total_time + time_diff))
        done
    fi

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
