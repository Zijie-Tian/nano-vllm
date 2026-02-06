# Nsys Profiling Rule

## 强制规则

**所有 nsys profiling 任务必须使用 `scripts/profile_offload.sh` 脚本**，禁止直接运行 nsys 命令。

| 禁止 | 原因 |
|------|------|
| `nsys profile python tests/test_ruler.py ...` | 参数不一致，输出路径混乱 |
| 手动构造 nsys 命令 | 容易遗漏关键参数 |

---

## ⚠️ GPU 指定方式 (CRITICAL)

**必须使用 `--gpu` 参数指定 GPU，禁止外部设置 `CUDA_VISIBLE_DEVICES`**

脚本内部会自己设置 `CUDA_VISIBLE_DEVICES`，外部设置会被覆盖！

```bash
# ✅ 正确：使用 --gpu 参数
bash scripts/profile_offload.sh --gpu 4 --policy xattn --ctx-len 32k

# ❌ 错误：外部设置 CUDA_VISIBLE_DEVICES 会被脚本覆盖
CUDA_VISIBLE_DEVICES=4 bash scripts/profile_offload.sh --gpu 0 ...
```

---

## 使用方法

### 基本用法

```bash
# 默认配置（GPU 0, full attention, 64k context）
bash scripts/profile_offload.sh

# 指定 GPU
bash scripts/profile_offload.sh --gpu 4

# 指定 sparse policy
bash scripts/profile_offload.sh --policy xattn

# 指定 context length
bash scripts/profile_offload.sh --ctx-len 128k
```

### 完整示例

```bash
# XAttention offload profiling on GPU 4, 32k context, GLM model
bash scripts/profile_offload.sh \
    --policy xattn \
    --ctx-len 32k \
    --gpu 4 \
    --model ~/models/GLM-4-9B-Chat-1M

# Full attention offload profiling on GPU 5, 64k context, Llama model
bash scripts/profile_offload.sh \
    --policy full \
    --ctx-len 64k \
    --gpu 5 \
    --model ~/models/Llama-3.1-8B-Instruct

# GPU-only mode (no offload) for comparison
bash scripts/profile_offload.sh \
    --policy xattn \
    --ctx-len 32k \
    --gpu 4 \
    --no-offload
```

### 并行测试示例

在多 GPU 上并行执行不同 context length 的 profiling：

```bash
# GPU 4: 32k, 128k, 512k
bash scripts/profile_offload.sh --policy xattn --ctx-len 32k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 128k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 512k --gpu 4 --model ~/models/GLM-4-9B-Chat-1M

# GPU 5: 64k, 256k, 768k (可与 GPU 4 并行执行)
bash scripts/profile_offload.sh --policy xattn --ctx-len 64k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 256k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
bash scripts/profile_offload.sh --policy xattn --ctx-len 768k --gpu 5 --model ~/models/GLM-4-9B-Chat-1M
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | `0` | **GPU ID（必须用此参数指定 GPU）** |
| `--policy` | `full` | Sparse policy: `full`, `xattn` |
| `--ctx-len` | `64k` | Context length: `32k`, `64k`, `128k`, `256k`, `512k`, `768k`, `1m` |
| `--model` | (auto) | 模型路径，默认使用 Llama-3.1-8B-Instruct |
| `--dataset` | `niah_single_1` | RULER 任务名称 |
| `--sample` | `0` | 样本索引 |
| `--num-gpu-blocks` | `4` | GPU ring buffer slots 数量 |
| `--block-size` | `4096` | KV cache block size |
| `--no-offload` | - | 禁用 CPU offload（GPU-only 模式） |

## 输出文件

输出文件自动生成到 `results/nsys/` 目录：

```
results/nsys/ruler_<dataset>_sample<index>_offload_<slots>slots_<timestamp>.nsys-rep
```

示例：`ruler_niah_single_1_sample0_offload_8slots_20260127_031500.nsys-rep`

## 查看结果

```bash
# GUI 查看
nsight-sys results/nsys/<filename>.nsys-rep

# 命令行统计
nsys stats --report cuda_api_sum results/nsys/<filename>.nsys-rep
nsys stats --report cuda_gpu_kern_sum results/nsys/<filename>.nsys-rep
```

## 典型工作流

### 1. 对比不同 slots 数量

```bash
# 测试 4 slots（默认）
bash scripts/profile_offload.sh --num-gpu-blocks 4

# 测试 8 slots
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 对比结果
nsys stats --report cuda_gpu_kern_sum results/nsys/*4slots*.nsys-rep
nsys stats --report cuda_gpu_kern_sum results/nsys/*8slots*.nsys-rep
```

### 2. 分析 pipeline overlap

```bash
# 生成 profile
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 用 nsight-sys GUI 查看 CUDA HW timeline
# 检查 H2D 和 flash_fwd_kernel 是否 overlap
```
