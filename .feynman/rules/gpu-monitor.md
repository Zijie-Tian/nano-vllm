# GPU Memory Monitoring Rule

## 强制规则

**所有 GPU 内存监控任务必须使用 `gpu-monitor` agent**，禁止使用以下方式：

| ❌ 禁止 | 原因 |
|--------|------|
| `nvidia-smi` 循环 + sleep | 阻塞主 agent，无法并行 |
| 后台 bash 监控脚本 | 难以管理，输出混乱 |
| 手动轮询 | 效率低，占用 context |

## 使用方法

```python
# 启动 GPU 监控（后台运行）
Task(
    subagent_type="gpu-monitor",
    prompt="Monitor GPU 0 with 0.5 second interval",
    run_in_background=True
)
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| GPU ID | 要监控的 GPU | `GPU 0`, `GPU 0,1` |
| interval | 采样间隔 | `0.5 second`, `1 second` |
| 目的 | 监控原因 | `for RULER benchmark test` |

## 典型用法

### 1. 单 GPU 基准测试
```
Monitor GPU 0 with 1 second interval for benchmark profiling
```

### 2. 调试 OOM
```
Monitor GPU 0 with 0.5 second interval to track memory peak during inference
```

### 3. 多 GPU 训练
```
Monitor GPU 0,1,2,3 with 2 second interval during training
```

## 获取结果

监控结果自动写入 output_file，使用以下方式读取：

```bash
# 查看最新输出
tail -50 /tmp/claude/.../tasks/<agent_id>.output

# 查找峰值
grep -i "peak\|max" /tmp/claude/.../tasks/<agent_id>.output
```

## 与测试并行

gpu-monitor 在后台运行，不会阻塞测试：

```python
# 1. 启动监控（后台）
Task(subagent_type="gpu-monitor", ..., run_in_background=True)

# 2. 运行测试（前台）
Bash("python tests/test_ruler.py ...")

# 3. 测试完成后查看监控结果
Bash("tail -50 <output_file>")
```
