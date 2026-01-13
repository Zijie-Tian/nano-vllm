# Progress Log

## Session: 2026-01-14

### Phase 0: 问题发现与分析
- **Status:** complete
- **Started:** 2026-01-14 02:00
- Actions taken:
  - 添加 nanovllm 作为 git submodule (tzj/vs_offload 分支)
  - 修改 run_ruler.sh 添加 PYTHONPATH 优先级
  - 4K 序列长度测试成功
  - 32K 序列长度测试失败 (OOM)
  - 分析 OOM 根因：每任务独立进程导致内存累积
  - 创建 planning-with-files 计划文档
- Files created/modified:
  - `.gitmodules` (添加 nanovllm submodule)
  - `scripts/run_ruler.sh` (PYTHONPATH 修改)
  - `eval/RULER/scripts/config_models.sh` (SEQ_LENGTHS 配置)
  - `task_plan.md` (本文档)
  - `findings.md` (发现记录)
  - `progress.md` (进度日志)

### Phase 1: 修改 run.sh 支持 nanovllm 多 GPU 并行
- **Status:** complete
- **Completed:** 2026-01-14 04:00
- Actions taken:
  - 添加 NUM_GPUS 环境变量（默认 4）
  - 检测 `MODEL_FRAMEWORK == "nanovllm"` 启用并行模式
  - 实现 round-robin GPU 分配 (1 task per GPU)
  - 每轮等待所有 GPU 完成后再启动下一批
  - 保持其他 backend 原有串行逻辑
- Files modified:
  - `eval/RULER/scripts/run.sh`

### Phase 3: 测试阶段一（单 GPU 批量任务）
- **Status:** pending
- Actions taken:
  -

### Phase 4: 测试阶段二（多 GPU 并行）
- **Status:** pending
- Actions taken:
  -

### Phase 5: 32K 全任务验证
- **Status:** pending
- Actions taken:
  -

---

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 4K nanovllm 单任务 | niah_single_1 | 成功 | 成功 | ✓ |
| 32K nanovllm 单任务 | niah_single_1 | 成功 | OOM | ✗ |
| 32K nanovllm 多任务 | - | - | 待测试 | - |
| 32K 4GPU 并行 | - | - | 待测试 | - |

---

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-01-14 02:42 | CUDA OOM | 1 | 发现根因，设计两阶段方案 |

---

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 完成，准备测试 |
| Where am I going? | Phase 3-5: 测试验证 |
| What's the goal? | 32K NanoVLLM 测试通过 + 4x 性能提升 |
| What have I learned? | 简化方案：只改 run.sh，1 task/GPU |
| What have I done? | 修改 run.sh 实现 nanovllm 并行执行 |

---

*Update after completing each phase or encountering errors*
