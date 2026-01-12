# RULER Migration Progress Log

## Session: 2026-01-13

### Time: Session Start
- [x] 初始化 COMPASS git 仓库
- [x] 设置 remote origin
- [x] 切换默认分支到 main
- [x] 初始提交并推送

### Time: 代码分析阶段
- [x] 探索 x-attention/eval/RULER 目录结构
- [x] 分析 scripts/run_ruler_*.sh 入口脚本
- [x] 阅读 eval/RULER/scripts/run.sh 主流程
- [x] 分析 config_models.sh 模型配置
- [x] 分析 config_tasks.sh 任务配置
- [x] 阅读 pred/call_api.py (关键 - xattn 依赖)
- [x] 阅读 pred/model_wrappers.py (关键 - xattn 依赖)
- [x] 阅读 data/prepare.py 数据准备流程
- [x] 阅读 data/synthetic/niah.py 数据生成
- [x] 阅读 eval/evaluate.py 评估流程
- [x] 分析 xattn 包结构和依赖

### Time: 规划文档创建
- [x] 创建 task_plan.md - 完整迁移计划
- [x] 创建 findings.md - 调用链和依赖分析
- [x] 创建 progress.md - 进度日志

---

## 待执行任务 (Next Steps)

### Phase 1: 创建项目目录结构
- [ ] 创建 compass/ 包目录及 __init__.py
- [ ] 创建 compass/src/ 子目录
- [ ] 创建 compass/threshold/ 子目录
- [ ] 创建 eval/RULER/scripts/ 目录结构
- [ ] 创建 scripts/ 顶层脚本目录

### Phase 2: 迁移核心算子模块
- [ ] 复制 xattn/src/*.py → compass/src/
- [ ] 复制 xattn/threshold/*.py → compass/threshold/
- [ ] 修改所有 `from xattn.` → `from compass.`

### Phase 3: 迁移 RULER Benchmark
- [ ] 复制 eval/RULER/scripts/run.sh
- [ ] 复制 eval/RULER/scripts/*.sh 配置脚本
- [ ] 复制 eval/RULER/scripts/synthetic.yaml
- [ ] 复制 eval/RULER/scripts/data/ 目录
- [ ] 复制 eval/RULER/scripts/pred/ 目录
- [ ] 复制 eval/RULER/scripts/eval/ 目录
- [ ] 修改 pred/call_api.py 的 import
- [ ] 修改 pred/model_wrappers.py 的 import

### Phase 4: 迁移顶层脚本
- [ ] 复制并修改 scripts/run_ruler_tasks.sh
- [ ] 复制并修改 scripts/run_ruler_docker.sh
- [ ] 复制并修改 scripts/run_ruler_nanovllm.sh

### Phase 5: 迁移配置文件
- [ ] 复制 eval/RULER/requirements.txt
- [ ] 复制 eval/RULER/Dockerfile
- [ ] 复制数据文件 (PaulGrahamEssays.json 等)

### Phase 6: 验证测试
- [ ] Python import 测试
- [ ] RULER 数据准备测试
- [ ] 模型推理测试
- [ ] 完整流程测试

---

## Files Modified/Created

| File | Action | Status |
|------|--------|--------|
| `README.md` | Created | Done |
| `task_plan.md` | Created | Done |
| `findings.md` | Created | Done |
| `progress.md` | Created | Done |

---

## Issues Encountered

*None so far*

---

## Notes

1. 项目从 x-attention 迁移到 COMPASS
2. 核心包名从 xattn 改为 compass
3. 关键外部依赖：flashinfer, block_sparse_attn, nemo-toolkit
4. nano-vllm 作为可选推理引擎，通过 PYTHONPATH 引入
5. Docker 镜像 `tzj/ruler:v0.3` 已包含必要依赖
