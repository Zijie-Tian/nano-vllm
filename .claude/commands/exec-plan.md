---
allowed-tools: Bash(CUDA_VISIBLE_DEVICES=*), Bash(PYTHONPATH=*), Bash(python*), Bash(git*), Bash(rm*), Bash(ls*), Bash(cat*), Bash(nvidia-smi*), Read, Edit, Write, Glob, Grep, TodoWrite, Task
argument-hint: --gpu <id> [--no-interrupt]
description: Execute task_plan.md refactoring with specified GPU, optionally without user interruption
---

# Execute Task Plan (exec-plan)

按照 `task_plan.md` 的要求执行代码重构，确保计划中的最终目标圆满实现。

## 参数说明

命令格式: `/exec-plan --gpu <id> [--no-interrupt]`

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gpu <id>` | **必需**。指定可用的 GPU ID，只能使用此 GPU 进行调试 | `--gpu 0`, `--gpu 2` |
| `--no-interrupt` | 可选。禁止中断执行，遇到问题不与用户交互，自动解决或跳过 | `--no-interrupt` |

## 当前参数

```
$ARGUMENTS
```

## 执行前准备

### 1. 解析参数

从 `$ARGUMENTS` 中解析：
- `GPU_ID`: 从 `--gpu <id>` 或 `-g <id>` 提取
- `NO_INTERRUPT`: 是否存在 `--no-interrupt` 或 `-n` 标志

### 2. 参数验证

**必须验证**:
- GPU_ID 必须是有效的数字
- 运行 `nvidia-smi -i <GPU_ID>` 验证 GPU 存在

### 3. 读取 task_plan.md

读取项目根目录下的 `task_plan.md` 文件，理解：
- 总体目标
- 分阶段计划 (Phase 1, 2, 3...)
- 文件修改清单
- 风险和注意事项
- 测试计划

## 执行流程

### Step 1: 创建执行计划

使用 TodoWrite 工具创建详细的执行计划，包括：
- 从 task_plan.md 提取的所有 Phase
- 每个 Phase 的子任务
- 测试验证步骤

### Step 2: 按 Phase 执行重构

对于 task_plan.md 中的每个 Phase：

1. **读取当前代码**: 使用 Read/Grep 理解现有实现
2. **实施修改**: 使用 Edit/Write 进行代码修改
3. **验证修改**: 运行相关测试

### Step 3: 运行测试验证

执行 task_plan.md 中定义的测试计划，验证重构成功。

## GPU 限制规则

**严格限制**: 只能使用指定的 GPU，所有涉及 GPU 的命令必须加 `CUDA_VISIBLE_DEVICES` 前缀：

```bash
# 正确
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python test.py

# 错误 - 禁止使用其他 GPU
python test.py  # 可能使用默认 GPU 0
CUDA_VISIBLE_DEVICES=0,1 python test.py  # 使用多个 GPU
```

## 中断模式规则

### 当 `--no-interrupt` 生效时

遇到以下情况**不停下来询问用户**，而是：

| 情况 | 处理方式 |
|------|----------|
| 测试失败 | 记录失败原因，尝试自动修复，继续下一步 |
| 代码冲突 | 尝试合理解决，记录解决方案 |
| 不确定的实现细节 | 选择最合理的方案继续 |
| 执行错误 | 分析错误，尝试修复，记录问题 |

**自动决策原则**:
1. 优先保证功能正确性
2. 遵循现有代码风格
3. 选择简单直接的实现
4. 记录所有自动决策到 `progress.md`

### 当未指定 `--no-interrupt` 时

遇到以下情况**可以询问用户**：
- 多个实现方案需要选择
- 测试持续失败无法自动修复
- 发现 task_plan.md 中的问题或矛盾

## 执行记录

### 进度文件: progress.md

实时更新 `progress.md` 记录：

```markdown
## 执行进度

### Phase X: [名称]
- 状态: [进行中/完成/失败]
- 开始时间: [时间]
- 完成时间: [时间]
- 修改文件: [文件列表]
- 自动决策: [如果有]
- 问题记录: [如果有]
```

### 发现记录: findings.md

记录执行过程中的重要发现到 `findings.md`。

## 示例用法

```bash
# 使用 GPU 2，允许中断
/exec-plan --gpu 2

# 使用 GPU 0，不中断执行
/exec-plan --gpu 0 --no-interrupt

# 简短形式
/exec-plan -g 1 -n
```

## 完成标准

执行完成后，确保：

1. **所有 Phase 完成**: task_plan.md 中的所有 Phase 都已实施
2. **测试通过**: task_plan.md 中的测试计划全部通过
3. **代码质量**: 修改符合项目代码规范
4. **文档更新**: progress.md 包含完整执行记录

## 重要约束

1. **GPU 隔离**: 绝对不能使用指定 GPU 以外的设备
2. **遵循计划**: 严格按照 task_plan.md 执行，不做计划外的修改
3. **渐进式修改**: 每个 Phase 完成后验证，而不是最后一起验证
4. **回滚准备**: 重大修改前考虑是否需要 git commit 保存点
