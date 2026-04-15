# Pre-Plan Git Remote Check

## Purpose

在开始执行任何实现计划之前，检查远端是否有新的提交。避免在本地完成修改后发现需要处理合并冲突。

---

## Rule

### 触发时机

在以下情况下 **必须** 执行远端检查：

| 场景 | 必须检查 |
|------|---------|
| 进入 Plan Mode (`EnterPlanMode`) | ✅ |
| 开始实现新功能 | ✅ |
| 开始修复 bug | ✅ |
| 开始重构代码 | ✅ |
| 简单的文档更新 | ❌ (可选) |
| 只读操作（分析、研究） | ❌ |

### 检查命令

```bash
# 1. Fetch 远端最新状态（不修改本地代码）
git fetch origin

# 2. 检查当前分支是否落后于远端
git status -uno
# 或更精确地：
git rev-list HEAD..origin/$(git branch --show-current) --count
```

### 检查结果处理

| 状态 | 行动 |
|------|------|
| 本地与远端同步 | ✅ 继续执行计划 |
| 本地领先远端 | ✅ 继续执行计划（可提醒 push） |
| **本地落后远端** | ⚠️ **停止，询问用户** |
| 存在分叉 (diverged) | ⚠️ **停止，询问用户** |

---

## 当检测到远端有更新时

### 必须输出的信息

```markdown
## ⚠️ 检测到远端有新提交

**当前分支**: `branch-name`
**远端状态**: 领先本地 X 个提交

### 远端新提交
```
git log --oneline HEAD..origin/branch-name
```

### 请选择处理方式：
1. **Pull (merge)**: `git pull origin branch-name`
2. **Pull (rebase)**: `git pull --rebase origin branch-name`
3. **忽略**: 继续执行计划（可能需要后续处理冲突）
4. **中止**: 暂停计划，手动处理
```

### 必须等待用户响应

- **不要** 自动执行 pull 或 rebase
- **不要** 假设用户的偏好
- **必须** 等待用户明确指示后再继续

---

## 实现示例

### 在 Plan Mode 开始时的检查流程

```bash
# Step 1: Fetch
git fetch origin 2>/dev/null

# Step 2: Get current branch
BRANCH=$(git branch --show-current)

# Step 3: Check if behind
BEHIND=$(git rev-list HEAD..origin/$BRANCH --count 2>/dev/null || echo "0")

# Step 4: Check if ahead
AHEAD=$(git rev-list origin/$BRANCH..HEAD --count 2>/dev/null || echo "0")

# Step 5: Determine status
if [ "$BEHIND" -gt 0 ] && [ "$AHEAD" -gt 0 ]; then
    echo "DIVERGED: local +$AHEAD, remote +$BEHIND"
elif [ "$BEHIND" -gt 0 ]; then
    echo "BEHIND: remote has $BEHIND new commits"
elif [ "$AHEAD" -gt 0 ]; then
    echo "AHEAD: local has $AHEAD unpushed commits"
else
    echo "SYNCED"
fi
```

---

## 快速检查脚本

可以直接运行的一行命令：

```bash
git fetch origin && git status -sb | head -1
```

输出解读：
- `## branch-name` - 已同步
- `## branch-name...origin/branch-name [ahead 2]` - 本地领先
- `## branch-name...origin/branch-name [behind 3]` - 本地落后
- `## branch-name...origin/branch-name [ahead 2, behind 3]` - 分叉

---

## 例外情况

以下情况可以跳过检查：

1. **用户明确要求跳过**: "忽略远端检查，直接开始"
2. **离线工作**: 无法连接到远端仓库
3. **新分支**: 本地分支在远端不存在
4. **只读任务**: 仅分析代码，不做修改

---

## Checklist

进入 Plan Mode 之前：

- [ ] 执行 `git fetch origin`
- [ ] 检查当前分支是否落后远端
- [ ] 如果落后或分叉，输出警告并等待用户指示
- [ ] 用户确认后再继续执行计划
