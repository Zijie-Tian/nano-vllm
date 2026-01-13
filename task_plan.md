# Task Plan: RULER + NanoVLLM 多 GPU 并行测试方案

## Goal
实现 RULER 测试框架对 NanoVLLM 后端的多 GPU 并行支持，解决 32K 上下文测试的 OOM 问题，并提升测试效率 4 倍。

## Current Status: `in_progress`
- Start Date: 2026-01-14

---

## 问题定义

**核心问题**：nanovllm 只支持单 GPU，RULER 每个任务独立调用 `call_api.py`，导致：
1. GPU 内存无法完全释放 → OOM
2. 无法利用多 GPU 并行加速
3. 每次重新加载模型，浪费时间

**约束**：不修改 nanovllm 内部代码，只从 RULER 测试端解决

---

## Phases

### Phase 1: 修改 run.sh 支持 nanovllm 多 GPU 并行 `complete`

**修改文件**: `eval/RULER/scripts/run.sh`

**简化方案**（仅修改 run.sh）:
- 检测 `MODEL_FRAMEWORK == "nanovllm"` 时启用并行模式
- 一个 task 一个 GPU，轮询分配 (`GPU_ID = task_index % NUM_GPUS`)
- 每轮等待所有 GPU 完成后再启动下一批
- 其他 backend 保持原有串行逻辑

**核心改动**:
```bash
# NanoVLLM: parallel execution (1 task per GPU, round-robin)
if [ "$MODEL_FRAMEWORK" == "nanovllm" ]; then
    for TASK in "${TASKS[@]}"; do
        GPU_ID=$((TASK_INDEX % NUM_GPUS))
        CUDA_VISIBLE_DEVICES=${GPU_ID} python pred/call_api.py --task ${TASK} ... &
        PIDS+=($!)
        # Wait when all GPUs are occupied
        if [ $((TASK_INDEX % NUM_GPUS)) -eq 0 ]; then
            wait "${PIDS[@]}"
        fi
    done
fi
```

**任务清单**:
- [x] 添加 NUM_GPUS 环境变量（默认 4）
- [x] 检测 nanovllm 后端启用并行
- [x] 实现 round-robin GPU 分配
- [x] 保持其他 backend 串行逻辑

---

### ~~Phase 2: 修改 run.sh 支持多 GPU 并行~~ (已合并到 Phase 1)

**修改文件**: `eval/RULER/scripts/run.sh`

**核心改动**:
```bash
# 添加并行模式参数
PARALLEL_MODE=${PARALLEL_MODE:-false}
NUM_GPUS=${NUM_GPUS:-4}

if [ "$PARALLEL_MODE" = "true" ]; then
    # 任务分组
    TASK_GROUPS=(
        "niah_single_1,niah_single_2,niah_single_3"
        "niah_multikey_1,niah_multikey_2,niah_multikey_3"
        "niah_multivalue,niah_multiquery,vt"
        "cwe,fwe,qa_1,qa_2"
    )

    # 并行启动
    for i in "${!TASK_GROUPS[@]}"; do
        GPU_ID=$((i % NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$GPU_ID python pred/call_api.py \
            --task "${TASK_GROUPS[$i]}" ... &
    done
    wait
else
    # 原有串行逻辑
    for TASK in "${TASKS[@]}"; do
        python pred/call_api.py --task "$TASK" ...
    done
fi
```

**任务清单**:
- [ ] 添加 PARALLEL_MODE 和 NUM_GPUS 参数
- [ ] 实现任务分组逻辑（13 任务 → 4 组）
- [ ] 添加多进程并行调度代码
- [ ] 保持原有串行模式的兼容性

---

### Phase 3: 测试阶段一（单 GPU 批量任务） `pending`

**验证方法**:
```bash
# 测试多任务支持
CUDA_VISIBLE_DEVICES=0 python pred/call_api.py \
    --task "niah_single_1,niah_single_2" \
    --server_type nanovllm \
    --model_name_or_path ~/models/Llama-3.1-8B-Instruct

# 预期：两个任务成功完成，无 OOM
```

**任务清单**:
- [ ] 在 GPU 0 上测试多任务支持
- [ ] 验证 LLM 实例复用正常工作
- [ ] 确认无 OOM 错误

---

### Phase 4: 测试阶段二（多 GPU 并行） `pending`

**验证方法**:
```bash
# 测试 4 GPU 并行
PARALLEL_MODE=true NUM_GPUS=4 \
    ./run.sh llama3.1-8b-nanovllm synthetic --metric full

# 预期：4 个 GPU 同时工作，总时间约为串行的 1/4
```

**任务清单**:
- [ ] 使用 4 GPU 运行并行测试
- [ ] 验证任务分组执行正确
- [ ] 确认 4x 性能提升

---

### Phase 5: 32K 全任务验证 `pending`

**验证方法**:
```bash
# 修改 SEQ_LENGTHS 为 32768
# 运行完整测试
./run.sh llama3.1-8b-nanovllm synthetic --metric full

# 预期：所有 13 个任务成功完成
```

**任务清单**:
- [ ] 配置 SEQ_LENGTHS 为 32768
- [ ] 运行所有 13 个任务（sample=2）
- [ ] 验证全部通过

---

## 关键文件

| 文件 | 修改内容 |
|------|----------|
| `eval/RULER/scripts/pred/call_api.py` | 批量任务支持、LLM 复用 |
| `eval/RULER/scripts/run.sh` | 多 GPU 并行调度 |
| `eval/RULER/scripts/config_tasks.sh` | 任务分组配置（可选） |

---

## Key Questions
1. 不同任务的 `tokens_to_generate` 是否差异很大？（需要取最大值创建 LLM）
2. 13 个任务如何合理分组到 4 个 GPU？（考虑任务复杂度平衡）
3. 任务失败时是否需要重试机制？

---

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 不修改 nanovllm 内部代码 | 用户约束：从 RULER 测试端解决问题 |
| 采用两阶段实现 | 阶段一解决 OOM，阶段二提升性能，降低风险 |
| LLM 实例复用而非热重载 | 简单可靠，避免复杂的模型状态管理 |
| 任务分 4 组对应 4 GPU | 匹配硬件配置（4x RTX 3090） |

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| CUDA OOM at 32K | 1 | 发现根因：每个任务独立创建 LLM，内存未释放 |

---

## 预期效果

| 指标 | 修改前 | 阶段一后 | 阶段二后 |
|------|--------|----------|----------|
| 32K OOM | ❌ 失败 | ✅ 成功 | ✅ 成功 |
| GPU 利用率 | 25% | 25% | 100% |
| 测试时间 | ~160s | ~100s | ~40s |

---

## Notes
- 核心问题：nanovllm 单 GPU + RULER 每任务独立进程 = 内存泄漏
- 解决思路：单进程内复用 LLM，多进程跨 GPU 并行
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
