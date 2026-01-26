# Nsys Profiling Rule

## 强制规则

**所有 nsys profiling 任务必须使用 `scripts/profile_offload.sh` 脚本**，禁止直接运行 nsys 命令。

| 禁止 | 原因 |
|------|------|
| `nsys profile python tests/test_ruler.py ...` | 参数不一致，输出路径混乱 |
| 手动构造 nsys 命令 | 容易遗漏关键参数 |

## 使用方法

```bash
# 基本用法（默认 4 slots）
bash scripts/profile_offload.sh

# 指定 GPU slots 数量
bash scripts/profile_offload.sh --num-gpu-blocks 8

# 指定 sample
bash scripts/profile_offload.sh --sample 5

# 指定 dataset
bash scripts/profile_offload.sh --dataset niah_single_1

# 禁用 offload（对比测试）
bash scripts/profile_offload.sh --no-offload

# 组合参数
bash scripts/profile_offload.sh --num-gpu-blocks 8 --sample 0 --gpu 1
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `niah_single_1` | RULER 任务名称 |
| `--sample` | `0` | 样本索引 |
| `--gpu` | `0` | 使用的 GPU |
| `--num-gpu-blocks` | `4` | GPU ring buffer slots 数量 |
| `--no-offload` | - | 禁用 CPU offload |

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
