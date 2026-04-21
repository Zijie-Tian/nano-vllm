# COMPASS RULER Benchmark 环境配置指南

本文档详细说明如何配置 COMPASS 项目的 RULER 基准测试环境。

## 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **CUDA**: 12.8+
- **GPU**: NVIDIA GPU (RTX 4090, A100, H100 等)
- **Python**: 3.10
- **Git**: 2.0+

## 快速配置

### 1. 创建 Conda 环境

```bash
# 创建名为 ruler 的 conda 环境
conda create -n ruler python=3.10 -y
conda activate ruler
```

### 2. 安装 PyTorch (CUDA 12.8)

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装基础依赖

```bash
# 进入项目根目录
cd /path/to/COMPASS

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt
```

### 4. 初始化 Git 子模块

```bash
# 添加 flash-attention 子模块 (使用 main 分支)
git submodule add https://github.com/Zijie-Tian/flash-attention.git 3rdparty/flash-attention

# 添加 flashinfer 子模块 (使用 tzj/minference 分支)
git submodule add -b tzj/minference https://github.com/Zijie-Tian/flashinfer.git 3rdparty/flashinfer

# 或者如果子模块已存在，只需更新
git submodule update --init --recursive
```

### 5. 编译 flash-attention

```bash
cd 3rdparty/flash-attention

# 初始化子模块 (cutlass 等)
git submodule update --init --recursive

# 编译安装
pip install -e . --no-build-isolation
```

**注意**: 编译可能需要 10-30 分钟，取决于硬件。

### 6. 编译 flashinfer

```bash
cd ../flashinfer  # 或 cd 3rdparty/flashinfer

# 初始化子模块
git submodule update --init --recursive

# 设置编译选项
export FLASHINFER_ENABLE_AOT=0

# 编译安装
pip install -e . --no-build-isolation
```

### 7. 验证环境

```bash
# 激活环境
conda activate ruler
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH

# 运行验证脚本
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import flash_attn
print(f'flash_attn: {flash_attn.__version__}')

import flashinfer
print('flashinfer: OK')

import transformers
print(f'transformers: {transformers.__version__}')

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
print('nemo manifest_utils: OK')

from compass.src.Fullprefill import Full_prefill
print('compass Full_prefill: OK')

print('=== Environment OK ===')
"
```

## 运行 RULER 测试

### 基本用法

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 指定 GPU
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH
export MODEL_DIR=/path/to/models  # 模型存放目录

# 进入脚本目录
cd /path/to/COMPASS/eval/RULER/scripts

# 运行测试
bash run.sh <MODEL_NAME> <BENCHMARK> [OPTIONS]
```

### 示例命令

```bash
# 测试 full attention (完整基准)
bash run.sh llama3.1-8b-chat synthetic --metric full

# 测试 xattn (X-attention)
bash run.sh llama3.1-8b-chat synthetic --metric xattn

# 测试 postrope (NanoVLLM POSTROPE policy)
./scripts/run_ruler.sh llama3.1-8b-nanovllm synthetic postrope

# 指定特定任务
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1

# 测试多个任务 (逗号分隔)
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1,niah_single_2,vt
```

### 可用的 Metric 选项

| Metric | 描述 |
|--------|------|
| `full` | 完整 attention (基准) |
| `postrope` | 显式 post-RoPE baseline policy |
| `xattn` | X-attention 优化 |
| `avgpool` | 平均池化稀疏 |
| `minfer` | Minference 优化 |
| `compass` | COMPASS 方法 |
| `flex` | FlexPrefill |

### 可用的 Task

```
niah_single_1, niah_single_2, niah_single_3
niah_multikey_1, niah_multikey_2, niah_multikey_3
niah_multivalue, niah_multiquery
vt, cwe, fwe
qa_1, qa_2
```

## 配置文件说明

### config_tasks.sh

控制测试任务和样本数量：

```bash
NUM_SAMPLES=5  # 每个任务的样本数 (调试用 2，完整测试用 5+)

synthetic=(
    "niah_single_1"
    "niah_single_2"
    # ... 其他任务
)
```

### config_models.sh

配置模型路径和参数，包含 `llama3.1-8b-chat` 等模型定义。

POSTROPE 的详细运行方式、配置驱动 workflow、以及已验证命令见：

- [`docs/POSTROPE_RULER_INTEGRATION.md`](POSTROPE_RULER_INTEGRATION.md)

## 常见问题

### Q: flash-attention 编译失败，提示 cutlass 缺失

```bash
cd 3rdparty/flash-attention
git submodule update --init --recursive
pip install -e . --no-build-isolation
```

### Q: transformers 版本兼容性问题

确保使用 transformers 4.53.3，已测试兼容。

### Q: Import 失败 (Xattention, Minference 等)

确保 PYTHONPATH 包含 COMPASS 根目录：
```bash
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH
```

### Q: GPU 内存不足

- 减少 batch_size (在 run.sh 中设置 BATCH_SIZE=1)
- 使用更短的序列长度 (修改 SEQ_LENGTHS)
- 使用稀疏 attention 方法 (xattn, avgpool 等)

## 目录结构

```
COMPASS/
├── compass/                 # 主代码包
│   ├── src/                 # 核心实现
│   │   ├── load_llama.py    # 模型加载和 FastPrefill
│   │   ├── Fullprefill.py   # Full attention
│   │   ├── Xattention.py    # X-attention
│   │   ├── AvgPool.py       # AvgPool 稀疏
│   │   ├── Compass.py       # COMPASS 方法
│   │   └── ...
│   └── threshold/           # 阈值配置
├── eval/
│   └── RULER/
│       └── scripts/         # RULER 测试脚本
│           ├── run.sh       # 主运行脚本
│           ├── config_tasks.sh
│           ├── config_models.sh
│           └── benchmark_root/  # 测试结果 (gitignored)
├── 3rdparty/
│   ├── flash-attention/     # flash-attention 子模块
│   └── flashinfer/          # flashinfer 子模块
├── docs/
│   └── ENVIRONMENT_SETUP.md # 本文档
├── requirements.txt         # Python 依赖
└── .gitignore
```

## 版本信息

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| transformers | 4.53.3 |
| flash_attn | 2.8.3 |
| flashinfer | 0.5.3 |
| nemo-toolkit | 2.6.1 |
| triton | 3.5.1 |
