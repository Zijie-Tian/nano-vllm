# Codex Configuration - COMPASS Project

## Scope & Source of Truth
- This `AGENTS.md` is the Codex entry point for the repository.
- `CLAUDE.md` remains the Claude-side entrypoint; `.codex/rules/*.md` is the complete Codex-side rule set, fully migrated from `.claude/rules/*.md`.
- `.codex/config.toml` is configured so Codex can fall back to `CLAUDE.md` and `GEMINI.md` in directories that do not have their own `AGENTS.md`.
- Thin Codex wrappers live under `.codex/skills/`; their source of truth for Codex is now `.codex/rules/`, so Codex no longer depends on `.claude/rules/`.

## 📚 Documentation References (Lazy Loading)

> **为减少 Codex runtime token 消耗，`docs/` 下的详细文档按需加载。需要时读取对应文档，不要默认全部展开。**

| 文档 | 路径 | 用途 | 何时读取 |
|------|------|------|----------|
| 环境配置指南 | `docs/ENVIRONMENT_SETUP.md` | 完整环境配置步骤、依赖版本、常见问题 | 配置新环境、检查依赖或排查环境问题时 |
| Claude 配置任务 | `docs/CLAUDE_SETUP_TASK.md` | Claude 侧环境初始化流程 | 对齐 Claude 环境、迁移到新主机时 |
| 可行性报告模板 | `docs/FEASIBILITY_REPORT_TEMPLATE.md` | 标准 feasibility report 输出格式 | 使用 `codex-deep-thinker` 做方案评估时 |
| XAttention chunked prefill | `docs/XATTN_CHUNKED_PREFILL.md` | Chunked prefill 实现、API、使用方式 | 修改或分析 `xattn_chunked` / chunked prefill 时 |
| RULER NanoVLLM XAttention 集成计划 | `docs/RULER_NANOVLLM_XATTN_INTEGRATION.md` | RULER 与 NanoVLLM XAttention 的集成背景与计划 | 处理 RULER + NanoVLLM + XAttention 集成时 |
| RULER XAttention 双路径分析 | `docs/RULER_NANOVLLM_XATTN_PATHS_ANALYSIS.md` | COMPASS 路径与 NanoVLLM 路径的差异分析 | 排查 XAttention 双实现路径、定位调用链时 |
| GQA dense head 问题分析 | `docs/GQA_DENSE_HEAD_ISSUE.md` | Chunked prefill + CPU offload 下的 dense head 瓶颈分析 | 分析 GQA / KV group 稀疏化瓶颈时 |
| Q head regrouping 方案 | `docs/04_q_head_regrouping.md` | Solution 4 方案设计与可行性结论 | 评估 dense head 重分组方案时 |
| KV hidden-dim reorder 方案 | `docs/05_kv_dim_reorder.md` | Solution 5 方案设计与可行性结论 | 评估 KV 维度重排方案时 |
| BLASST RULER 集成指南 | `docs/BLASST_RULER_INTEGRATION.md` | BLASST 在 RULER benchmark 中的运行与配置说明 | 使用或调试 BLASST metric 时 |
| LongBench Benchmark 指南 | `docs/LONG_BENCH_BENCHMARK_GUIDE.md` | LongBench 命令入口、模型别名、子集 preset 与输出规则 | 配置或运行 `run_longbench.sh` / `eval/LongBench/scripts/run.sh` 时 |
| TriAttention Calibration 指南 | `docs/TRIATTENTION_CALIBRATION_GUIDE.md` | 解释 `scripts/calibrate_triattention.py` 的统计对象、公式、离线校准意义与使用方式 | 理解或使用 TriAttention stats 标定时 |
| TriAttention 复现指南 | `docs/TRIATTENTION_REPRODUCTION_GUIDE.md` | 解释当前工程内 TriAttention 的复现逻辑、LongBench 接入方式，以及 standard 对照脚本如何使用 | 对比当前实现与 standard 实现、或复现实验时 |

### 使用方式
```bash
# 配置环境时，按需读取环境文档
sed -n '1,220p' docs/ENVIRONMENT_SETUP.md

# 做可行性分析前，先读取报告模板
sed -n '1,260p' docs/FEASIBILITY_REPORT_TEMPLATE.md

# 运行 LongBench 前，按需读取 LongBench 指南
sed -n '1,260p' docs/LONG_BENCH_BENCHMARK_GUIDE.md

# 理解 TriAttention 标定脚本时，按需读取 TriAttention Calibration 指南
sed -n '1,260p' docs/TRIATTENTION_CALIBRATION_GUIDE.md

# 理解当前 TriAttention 复现逻辑和 standard 对照脚本时，按需读取复现指南
sed -n '1,260p' docs/TRIATTENTION_REPRODUCTION_GUIDE.md
```

## Structured Codex Surfaces

| Surface | Path | Purpose |
|---------|------|---------|
| Project config | `.codex/config.toml` | Codex project overrides, fallback filenames, structured skill registration |
| Custom agent | `.codex/agents/codex-deep-thinker.toml` | Deep analysis, feasibility verification, architecture trade-offs |
| Structured skills | `.codex/skills/*/SKILL.md` | Codex wrappers that map to `.codex/rules/*.md` and other repo-local implementations |
| Codex rule set | `.codex/rules/*.md` | Full migrated rule set from `.claude/rules/*.md`, now consumed directly by Codex |
| Existing repo skill | `.agents/skills/ruler_results_summarizer/SKILL.md` | RULER results aggregation implementation reused by Codex |
| Shared agent memory | `.claude/agent-memory/codex-deep-thinker/MEMORY.md` | Shared long-lived memory for deep analysis work |

## Rule ↔ Skill Map

> 按任务类型优先命中对应 `.codex/skills/` skill；如果需要细节，再读取右侧对应的 `.codex/rules/*.md` 源文件。多个规则可以叠加使用。

| Area | Codex skill | Canonical source | When to use |
|------|-------------|------------------|-------------|
| Runtime commands | `.codex/skills/compass-commands/SKILL.md` | `.codex/rules/commands.md` | 运行脚本、benchmark、import check、环境初始化 |
| Code analysis | `.codex/skills/compass-code-analysis/SKILL.md` | `.codex/rules/code-analysis.md` | 查调用链、定位符号、理解实现、规划重构 |
| Testing | `.codex/skills/compass-testing/SKILL.md` | `.codex/rules/testing.md` | 新建/修改测试、设计验证脚本、解释测试风格 |
| GPU testing | `.codex/skills/compass-gpu-testing/SKILL.md` | `.codex/rules/gpu-testing.md` | 任何 GPU benchmark / test / profiling 命令 |
| 3rdparty protection | `.codex/skills/compass-3rdparty-policy/SKILL.md` | `.codex/rules/3rdparty-policy.md` | 触碰 `3rdparty/` 前 |
| Documentation management | `.codex/skills/compass-doc-management/SKILL.md` | `.codex/rules/doc-management.md` | 需要新增/改写技术文档或整理 `docs/` |
| Documentation lazy loading | `.codex/skills/compass-documentation-lazy-loading/SKILL.md` | `.codex/rules/documentation-lazy-loading.md` | 调整 `CLAUDE.md` / `AGENTS.md` / docs 索引结构 |
| No extra docs | `.codex/skills/compass-no-extra-docs/SKILL.md` | `.codex/rules/no-extra-docs.md` | 评估是否应该新建 markdown 文档时 |
| Feasibility report | `.codex/skills/compass-feasibility-report/SKILL.md` | `.codex/rules/feasibility-report.md` | 方案可行性分析、架构 trade-off、PoC 评估 |
| KVCache/RoPE data | `.codex/skills/compass-kvcache-rope-data/SKILL.md` | `.codex/rules/kvcache-rope-data.md` | 任何依赖 `results/kvcache-rope/` 数据的分析或脚本 |
| Git pre-plan check | `.codex/skills/compass-pre-plan-git-check/SKILL.md` | `.codex/rules/pre-plan-git-check.md` | 开始实现、修 bug、重构之前 |
| RULER task config | `.codex/skills/compass-ruler-task-config/SKILL.md` | `.codex/rules/ruler-task-config.md` | 修改 `eval/RULER/scripts/*task*` 相关配置 |
| AliyunPan download | `.codex/skills/aliyunpan/SKILL.md` | `.codex/rules/kvcache-rope-data.md` | 本地缺少网盘数据，需要下载时 |
| RULER result summary | `.codex/skills/compass-ruler-results-summarizer/SKILL.md` | `.agents/skills/ruler_results_summarizer/SKILL.md` | 汇总多个 `summary.csv` 的 benchmark 结果 |

## Repo-local Custom Agent
- Use `.codex/agents/codex-deep-thinker.toml` for deep analytical thinking, solution comparison, feasibility verification, and complex root-cause analysis.
- Behavior parity source: `.claude/agents/codex-deep-thinker.md`
- Shared memory source: `.claude/agent-memory/codex-deep-thinker/MEMORY.md`
- If the task is a feasibility evaluation, also read `.codex/rules/feasibility-report.md` and `docs/FEASIBILITY_REPORT_TEMPLATE.md` before drafting output.

## Project-Specific: COMPASS RULER Benchmark

### Running RULER
```bash
conda activate ruler
export PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH
export MODEL_DIR=/path/to/models
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd eval/RULER/scripts
bash run.sh llama3.1-8b-chat synthetic --metric full
```

### Available Metrics
| Metric | Description |
|--------|-------------|
| `full` | Full attention (baseline) |
| `xattn` | X-attention sparse |
| `xattn_chunked` | X-attention (chunked prefill) |
| `avgpool` | Average pooling sparse |
| `minfer` | Minference |
| `compass` | COMPASS method |
| `flex` | FlexPrefill |

### Run Specific Tasks
```bash
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1,vt,qa_1
```

## Mandatory Global Rules
- Before implementation / refactor / bug-fix work, run the pre-plan git remote check from `.codex/rules/pre-plan-git-check.md`.
- Use `PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH` instead of `pip install -e .`.
- Prefer Serena MCP for code navigation when available; otherwise prefer symbol-aware Codex/native MCP tools before grep/glob.
- Do not modify `3rdparty/` in place. Update dependencies in their dedicated development repos first, then sync the stable copy/submodule back here.
- Before any GPU command, require an explicit GPU id from the user and prefix the command with `CUDA_VISIBLE_DEVICES=...`.
- Do not proactively create markdown documentation. Only create/update docs when the user explicitly requests documentation or when an existing documentation workflow requires syncing docs after a change.
- Keep `CLAUDE.md` concise; put technical detail in `docs/` and update the `CLAUDE.md` + `GEMINI.md` indexes whenever a new doc is added.
- If a task needs KVCache/RoPE data, check the local file first and use the `aliyunpan` skill only if the data is missing.
- When modifying RULER task config, update both `eval/RULER/scripts/config_tasks.sh` and `eval/RULER/scripts/eval.sh`, keep the full task list, and disable entries with comments instead of deleting lines.
- When a feasibility evaluation is requested, use the `codex-deep-thinker` agent and follow the feasibility-report rule + template exactly.

## important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
When updating the `nanovllm` submodule, ALWAYS ensure it is checked out to the `tzj/minference` branch.
