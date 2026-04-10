# Planning with Files Rule

## Git 管理政策

**重要**：Planning 文件已从 Git 管理中排除，不会被提交。

### 已配置的 .gitignore 规则

```gitignore
# Planning-with-files temporary files
task_plan.md
findings.md
progress.md
task_plan_*.md
findings_*.md
progress_*.md
```

### 为什么排除这些文件

1. **临时性质**：计划文件是会话级别的临时文件，不应进入版本控制
2. **避免冲突**：多实例并行开发时，不同任务的计划文件会产生冲突
3. **保持仓库整洁**：这些文件只对当前任务有用，不需要历史记录

### 如果不小心已经 commit 了

```bash
# 从 git 中移除（保留本地文件）
git rm --cached task_plan.md findings.md progress.md
git commit -m "chore: remove planning files from git tracking"
```

---

## 自动清理旧计划文件

**重要**：每次开始新的复杂任务使用 planning-with-files 时，先删除旧的计划文件。

### 使用前执行以下命令

```bash
# 在项目根目录执行，删除旧的计划文件
cd /home/zijie/Code/nano-vllm
rm -f task_plan.md findings.md progress.md
rm -f task_plan_*.md findings_*.md progress_*.md
```

### 为什么需要这个规则

1. **避免混淆**：不同任务有不同计划，旧的计划文件会干扰新任务
2. **保持简洁**：只保留当前任务的计划文件
3. **自动清理**：无需手动检查文件内容，直接删除即可

### 使用 planning-with-files 的完整流程

```bash
# Step 1: 清理旧计划文件
rm -f task_plan.md findings.md progress.md

# Step 2: 启动 planning-with-files 技能
# 在 Claude 中调用 /planning-with-files 或 Skill tool

# Step 3: 技能会自动创建新的计划文件
# - task_plan.md (或 task_plan_<任务名>.md)
# - findings.md (或 findings_<任务名>.md)
# - progress.md (或 progress_<任务名>.md)
```

### 文件命名建议

| 场景 | 文件命名 | 示例 |
|------|----------|------|
| 通用任务 | task_plan.md, findings.md, progress.md | 临时调试任务 |
| 特定功能 | task_plan_<feature>.md | task_plan_xattn.md |
| Bug 修复 | task_plan_bug_<name>.md | task_plan_bug_offload.md |

### 注意事项

- 计划文件存储在**项目根目录**，不是技能目录
- 技能目录：`/home/zijie/.feynman/plugins/cache/planning-with-files/...`
- 项目目录：`/home/zijie/Code/nano-vllm/`
- 每个任务完成后，可以选择保留或删除计划文件
