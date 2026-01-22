# Testing

## Test Code Style

所有测试代码遵循以下风格：

### 文件结构

```python
"""
Test: [模块名称]

[简要说明测试内容和数据流]
"""
import torch
import sys
sys.path.insert(0, "/home/zijie/Code/nano-vllm")
from nanovllm.xxx import xxx

# ============================================================
# 参数配置
# ============================================================

param1 = value1  # 说明约束条件
param2 = value2

# ============================================================
# 构造输入
# ============================================================

input_tensor = ...  # 使用结构化数据便于验证

# ============================================================
# Step N: [操作名称]
# ============================================================

output = some_function(input_tensor, ...)

# 验证: [验证逻辑说明]
expected = ...
actual = output[...].item()
assert actual == expected, f"xxx: {actual} != {expected}"

print("test_xxx: PASSED")
```

### 核心原则

| 原则 | 说明 |
|------|------|
| **最小化 print** | 只在最后输出 `PASSED`，不打印中间结果 |
| **结构化数据** | 使用可预测的输入（全 1、偶奇交替等）便于手算验证 |
| **注释说明验证逻辑** | 在 assert 前用注释解释预期值的计算方式 |
| **分段用 `====`** | 用 `# ============` 分隔参数、输入、各步骤 |
| **assert 验证** | 用 assert 而不是 print 比较结果 |

### 输出规范

```python
# ✅ 正确
assert actual == expected, f"xxx: {actual} != {expected}"
print("test_xxx: PASSED")

# ❌ 错误
print(f"输出: {output}")
print(f"预期: {expected}, 实际: {actual}")
```

### 参数注释

```python
# ✅ 正确: 注释说明约束条件
seq_len = 512       # Triton 要求 seq_len >= stride * BLOCK_M
segment_size = 128  # 必须 >= block_size

# ❌ 错误: 无意义的注释
seq_len = 512  # 序列长度
```

### 验证逻辑注释

```python
# ✅ 正确: 解释计算过程
# 验证: 反对角线求和
# Q[奇]*K[偶] + Q[偶]*K[奇] = 2*1 + 1*2 = 4，共 stride/2 对
expected = (2*1 + 1*2) * (stride // 2) * head_dim

# ❌ 错误: 只写公式不解释
expected = 4 * 2 * 128
```

## Running Tests

```bash
# 运行单个测试
PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_xxx.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_xxx.py
```

## Benchmarks

```bash
python bench.py           # GPU benchmark
python bench_offload.py   # CPU offload benchmark
python bench_vllm.py      # vLLM comparison
```
