# Findings & Decisions

## Requirements
- NanoVLLM 后端只能使用单 GPU（内部未实现数据并行）
- RULER 测试需要运行 13 个任务：NIAH(9) + VT + CWE + FWE + QA(2)
- 目标序列长度：32768 tokens
- 可用硬件：4x RTX 3090 (24GB each)
- 只关注 `full` metric 的情况

---

## Research Findings

### RULER 框架结构
- **run.sh**: 主入口脚本，遍历任务列表，为每个任务调用 call_api.py
- **call_api.py**: 推理脚本，创建 LLM 实例并处理单个任务
- **model_wrappers.py**: 包含 NanoVLLMModel 类，封装 nanovllm 调用

### OOM 根因分析
```
RULER 执行流程：
run.sh → for task in tasks:
           python call_api.py --task $task  # 每次新进程

问题：
1. 每个任务创建新进程 → 新 LLM 实例
2. 进程结束但 GPU 内存未完全释放
3. 累积导致 OOM
```

### NanoVLLM 配置
```python
# model_wrappers.py:15-30
class NanoVLLMModel:
    def __init__(self, name_or_path, **generation_kwargs):
        self.llm = LLM(name_or_path, **llm_kwargs)
        # LLM 在初始化时加载到 GPU
```

### 任务配置
```yaml
# config_tasks.sh
NUM_SAMPLES=2
TASKS=(niah_single_1 niah_single_2 niah_single_3
       niah_multikey_1 niah_multikey_2 niah_multikey_3
       niah_multivalue niah_multiquery vt
       cwe fwe qa_1 qa_2)
```

### call_api.py 关键结构
```python
# Line 71: 参数解析
parser.add_argument("--task", type=str, required=True)

# Line 105+: 任务配置加载
config = tasks_customized.get(args.task)
config.update(tasks_base[config['task']])

# Line 150+: LLM 创建
if args.server_type == 'nanovllm':
    llm = NanoVLLMModel(...)

# Line 200+: 推理执行
for batch in batches:
    responses = llm.generate(batch)
```

---

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| call_api.py 支持逗号分隔多任务 | 最小改动，向后兼容 |
| 取所有任务 max(tokens_to_generate) | 确保 LLM 配置足够处理任何任务 |
| try-finally 保证内存清理 | 即使出错也释放 GPU 资源 |
| run.sh 添加 PARALLEL_MODE 开关 | 保持原有行为，新功能可选启用 |

---

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 32K 测试 OOM | 设计两阶段方案解决 |
| block_sparse_attn 模块缺失 | 忽略：只关注 full metric |

---

## Resources
- `eval/RULER/scripts/pred/call_api.py` - 主推理脚本 (Line 71, 105, 150, 200)
- `eval/RULER/scripts/run.sh` - 任务调度脚本
- `eval/RULER/scripts/pred/model_wrappers.py` - NanoVLLMModel 定义 (Line 15-30)
- `eval/RULER/scripts/config_tasks.sh` - 任务列表配置
- `eval/RULER/scripts/config_models.sh` - 模型和序列长度配置

---

## Visual/Browser Findings
- nvidia-smi 显示 4x RTX 3090 可用
- 4K 测试通过，32K 测试在第 2-3 个任务时 OOM

---

*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
