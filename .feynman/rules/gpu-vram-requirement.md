# GPU VRAM Requirement Rule

## GPU-only 模式显存要求

**强制规则**：执行 GPU-only 代码（不启用 CPU offload）时，**必须**在 40GB 及以上显存的 GPU 上进行测试。

### 检测方法

在运行 GPU-only 测试之前，**必须**先检查 GPU 显存：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

### GPU 分类

| GPU 型号 | 显存 | GPU-only 测试 |
|----------|------|---------------|
| A100 40GB | 40GB | ✅ 允许 |
| A100 80GB | 80GB | ✅ 允许 |
| H100 80GB | 80GB | ✅ 允许 |
| A6000 | 48GB | ✅ 允许 |
| RTX 3090 | 24GB | ❌ **禁止**（仅 offload 模式） |
| RTX 4090 | 24GB | ❌ **禁止**（仅 offload 模式） |

### 执行流程

1. **检测 GPU 显存**（必须）
2. **显存 >= 40GB**：继续执行 GPU-only 测试
3. **显存 < 40GB**：**停止**，提示用户：
   > "当前 GPU 显存为 XXX GB，不满足 GPU-only 模式的最低 40GB 要求。请使用 `--enable-offload` 参数启用 CPU offload 模式。"

### 代码示例

```python
# 在运行 GPU-only benchmark 之前
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
vram_mb = int(result.stdout.strip().split('\n')[0])
if vram_mb < 40000:  # 40GB = 40000MB
    raise RuntimeError(f"GPU VRAM ({vram_mb}MB) < 40GB. Use --enable-offload for this GPU.")
```

### 适用范围

| 脚本 | 适用此规则 |
|------|-----------|
| `bench.py` | ✅ 必须检查显存 |
| `bench_offload.py` | ❌ 不适用（始终使用 offload） |
| `tests/test_*.py --enable-offload` | ❌ 不适用 |
| `tests/test_*.py` (无 offload) | ✅ 必须检查显存 |
