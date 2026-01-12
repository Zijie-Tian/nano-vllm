# RULER Migration Progress Log

## Session: 2026-01-13

### 代码分析阶段 ✓
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
- [x] 阅读 x-attention/task_plan.md (环境配置计划)

### 规划文档创建 ✓
- [x] 创建 task_plan.md - 代码迁移计划 (不含环境配置)
- [x] 创建 findings.md - 调用链和依赖分析
- [x] 创建 progress.md - 进度日志

---

## 待执行任务 (Next Steps)

### Phase 1: 创建项目目录结构
- [ ] 创建 `compass/` 包目录及 `__init__.py`
- [ ] 创建 `compass/src/` 子目录
- [ ] 创建 `compass/threshold/` 子目录
- [ ] 创建 `eval/RULER/scripts/` 目录结构
- [ ] 创建 `scripts/` 顶层脚本目录

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
- [ ] 创建 scripts/run_ruler.sh

### Phase 5: 创建本地 manifest_utils (NeMo ASR Extra 替代)
- [ ] 创建 `eval/RULER/scripts/utils/__init__.py`
- [ ] 创建 `eval/RULER/scripts/utils/manifest_utils.py`
- [ ] 修改 call_api.py, evaluate.py, niah.py 等文件的 import
- [ ] 检查 tokenizer.py 是否需要修改（大部分已用 HuggingFace）

**说明**: NeMo 2.6.1 base 已安装，但 ASR extra 不可用，需本地实现 manifest_utils

### Phase 6: 验证测试
- [ ] Python import 测试
- [ ] RULER 数据准备测试
- [ ] 模型推理测试
- [ ] 完整流程测试

---

## Files Created

| File | Status |
|------|--------|
| `README.md` | Done |
| `task_plan.md` | Done |
| `findings.md` | Done |
| `progress.md` | Done |

---

## Notes

1. **环境配置由 `x-attention/task_plan.md` 管理**，本计划只涉及代码迁移
2. 环境包括：PyTorch 2.9.1, transformers 4.57.3, **nemo-toolkit 2.6.1 (base)**
3. 原 xattn 包重命名为 compass 包
4. 所有 `from xattn.` 改为 `from compass.`
5. **NeMo base 不含 ASR extra**，需本地实现 `manifest_utils.py`
6. 3rdparty 目录（flash-attention, flashinfer）由 x-attention task_plan 管理
