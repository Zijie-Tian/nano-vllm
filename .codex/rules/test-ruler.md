# test_ruler.py 使用规则

## 强制规则

**执行 `test_ruler.py` 前必须查阅文档**，禁止运行 `--help` 或猜测参数。

| 禁止 | 原因 |
|------|------|
| `python tests/test_ruler.py --help` | 浪费交互，文档已有完整说明 |
| 猜测参数格式 | 容易出错，降低效率 |

## 必读文档

**[`docs/test_ruler_usage_guide.md`](../docs/test_ruler_usage_guide.md)** - 包含：
- 完整参数说明
- 已验证的命令示例
- GPU 模式选择指南
- max-model-len 设置指南

## 快速参考

### 标准命令格式

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/<MODEL> \
    --data-dir tests/data/ruler_<CTX> \
    --datasets <TASK> \
    --num-samples <N> \
    --max-model-len <LEN> \
    --enable-offload \
    [--sparse-policy XATTN_BSA] \
    [--sparse-threshold 0.9]
```

### 常用参数速查

| 参数 | 用途 | 示例 |
|------|------|------|
| `--datasets` | 指定任务 | `niah_single_1,qa_1` |
| `--num-samples` | 样本数 | `1`, `10`, `0`(全部) |
| `--sample-indices` | 指定索引 | `0,5,10` |
| `--enable-offload` | CPU offload | RTX 3090 必须 |
| `--sparse-policy` | 稀疏策略 | `XATTN_BSA` |
| `--json-output` | JSON 输出 | 脚本使用 |
| `--quiet` | 安静模式 | 减少输出 |

### max-model-len 速查

| 数据目录 | max-model-len |
|---------|---------------|
| ruler_32k | 40960 |
| ruler_64k | 72000 |
| ruler_128k | 135000 |

### 常用命令模板

**32K Offload + XAttn**:
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 40960 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

**64K Offload + XAttn**:
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_64k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 72000 \
    --enable-offload \
    --sparse-policy XATTN_BSA
```

## 执行前检查清单

- [ ] 用户指定了 GPU？否则询问
- [ ] RTX 3090/4090？必须 `--enable-offload`
- [ ] data-dir 与 max-model-len 匹配？
- [ ] 需要 density 统计？添加 `--sparse-policy XATTN_BSA`
