# Claude 环境配置任务

本文档供 Claude Code 在新主机上自动配置 COMPASS RULER 环境使用。

## 前置条件检查

在开始配置前，Claude 应先验证以下条件：

```bash
# 1. 检查操作系统
uname -a  # 应为 Linux

# 2. 检查 CUDA 版本
nvcc --version  # 需要 CUDA 12.8+
nvidia-smi  # 确认 GPU 可用

# 3. 检查 Conda
conda --version  # 需要 Anaconda/Miniconda

# 4. 检查 Git
git --version  # 需要 Git 2.0+
```

如果任何条件不满足，应提示用户先安装对应组件。

---

## 配置步骤 (按顺序执行)

### Step 1: 创建 Conda 环境

```bash
# 创建 Python 3.10 环境 (必须是 3.10)
conda create -n ruler python=3.10 -y

# 激活环境
conda activate ruler

# 验证 Python 版本
python --version  # 应显示 Python 3.10.x
```

**注意**: Python 版本必须是 3.10，其他版本可能导致依赖兼容性问题。

---

### Step 2: 安装 PyTorch (CUDA 12.8)

```bash
# 安装 PyTorch 2.9.1 + CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**验证安装**:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

预期输出:
- PyTorch: 2.9.1+cu128
- CUDA available: True
- CUDA version: 12.8

---

### Step 3: 安装 Python 依赖

```bash
# 进入项目根目录
cd /path/to/COMPASS  # 替换为实际路径

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt
```

**关键依赖版本** (requirements.txt 中已指定):
- transformers==4.53.3
- nemo-toolkit[asr]==2.6.1
- triton==3.5.1
- einops==0.8.1

---

### Step 4: 初始化 Git 子模块

```bash
cd /path/to/COMPASS  # 项目根目录

# 检查子模块是否已存在
ls -la 3rdparty/

# 如果 3rdparty/flash-attention 不存在或为空:
git submodule add https://github.com/Zijie-Tian/flash-attention.git 3rdparty/flash-attention

# 如果 3rdparty/flashinfer 不存在或为空:
git submodule add -b tzj/minference https://github.com/Zijie-Tian/flashinfer.git 3rdparty/flashinfer

# 更新所有子模块 (包括嵌套子模块)
git submodule update --init --recursive
```

**重要**: flashinfer 必须使用 `tzj/minference` 分支，不是 main 分支。

---

### Step 5: 编译 flash-attention

**预计时间**: 10-30 分钟 (取决于 CPU 性能)

```bash
cd /path/to/COMPASS/3rdparty/flash-attention

# 确保子模块完整 (包含 cutlass)
git submodule update --init --recursive

# 编译安装
pip install -e . --no-build-isolation
```

**验证安装**:
```bash
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"
```

预期输出: `flash_attn: 2.8.3`

**常见错误处理**:
- 如果提示 `cutlass` 缺失: 重新执行 `git submodule update --init --recursive`
- 如果编译失败: 检查 `nvcc --version` 确保 CUDA 正确安装
- 如果内存不足: 设置 `MAX_JOBS=4` 限制并行编译数

---

### Step 6: 编译 flashinfer

**预计时间**: 10-20 分钟

```bash
cd /path/to/COMPASS/3rdparty/flashinfer

# 确保子模块完整
git submodule update --init --recursive

# 关键: 设置编译选项 (禁用 AOT 编译)
export FLASHINFER_ENABLE_AOT=0

# 编译安装
pip install -e . --no-build-isolation
```

**验证安装**:
```bash
python -c "import flashinfer; print('flashinfer: OK')"
```

**重要**:
- `FLASHINFER_ENABLE_AOT=0` 是必须的，否则编译会失败
- 不要与 flash-attention 同时编译，应顺序执行

---

### Step 7: 验证完整环境

```bash
# 设置 PYTHONPATH
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH

# 运行完整验证
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

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

print('')
print('=== Environment Setup Complete ===')
"
```

**预期完整输出**:
```
PyTorch: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8
GPU: [GPU名称]
flash_attn: 2.8.3
flashinfer: OK
transformers: 4.53.3
nemo manifest_utils: OK
compass Full_prefill: OK

=== Environment Setup Complete ===
```

---

### Step 8: 下载测试数据集

**注意**: 此步骤会下载约 50MB 的测试数据，如果数据已存在则自动跳过。

```bash
cd /path/to/COMPASS/eval/RULER
bash setup.sh
```

setup.sh 会自动：
- 安装下载脚本所需依赖 (html2text, beautifulsoup4)
- 检测数据是否已存在
- 下载缺失的数据文件

**下载的数据文件**:
| 文件 | 大小 | 用途 |
|------|------|------|
| `PaulGrahamEssays.json` | ~3MB | NIAH 任务的背景文本 |
| `hotpotqa.json` | ~46MB | QA 任务数据集 |
| `squad.json` | ~4MB | QA 任务数据集 |

**验证数据**:
```bash
ls -lh /path/to/COMPASS/eval/RULER/scripts/data/synthetic/json/*.json
```

预期输出应显示 3 个 json 文件。

---

## 运行 RULER 测试

环境配置完成后，运行测试验证:

```bash
# 激活环境
conda activate ruler

# 设置环境变量
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH
export MODEL_DIR=/path/to/models  # 模型存放目录
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定 GPU

# 进入脚本目录
cd /path/to/COMPASS/eval/RULER/scripts

# 运行完整测试
bash run.sh llama3.1-8b-chat synthetic --metric full

# 或运行单个任务测试
bash run.sh llama3.1-8b-chat synthetic --metric full --task niah_single_1
```

---

## 关键版本对照表

| 组件 | 版本 | 来源 | 备注 |
|------|------|------|------|
| Python | 3.10 | conda | 必须 3.10 |
| PyTorch | 2.9.1+cu128 | pip (cu128 index) | CUDA 12.8 |
| CUDA | 12.8+ | 系统安装 | nvcc 必须可用 |
| transformers | 4.53.3 | pip | 已测试兼容 |
| flash_attn | 2.8.3 | 源码编译 | 3rdparty/flash-attention |
| flashinfer | 0.5.3 | 源码编译 | tzj/minference 分支 |
| nemo-toolkit | 2.6.1 | pip | 用于 manifest_utils |
| triton | 3.5.1 | pip | 自定义 kernel |

---

## 故障排除

### 1. flash-attention 编译失败

**症状**: `cutlass not found` 或编译中断

**解决**:
```bash
cd 3rdparty/flash-attention
git submodule update --init --recursive
pip install -e . --no-build-isolation
```

### 2. flashinfer 编译失败

**症状**: AOT 相关错误

**解决**:
```bash
export FLASHINFER_ENABLE_AOT=0
pip install -e . --no-build-isolation
```

### 3. Import 错误

**症状**: `ModuleNotFoundError: No module named 'compass'`

**解决**:
```bash
export PYTHONPATH=/path/to/COMPASS:$PYTHONPATH
```

### 4. CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 False

**解决**:
```bash
# 检查 CUDA 安装
nvcc --version
nvidia-smi

# 确保 PyTorch 是 CUDA 版本
pip show torch | grep Version  # 应显示 +cu128
```

### 5. GPU 内存不足

**症状**: CUDA out of memory

**解决**:
- 减少 batch_size
- 使用更少的 GPU: `export CUDA_VISIBLE_DEVICES=0`
- 使用稀疏 attention 方法: `--metric xattn`

### 6. transformers API 兼容性

**症状**: `ValueError: too many values to unpack`

**说明**: 代码已针对 transformers 4.53.3 修复。如使用其他版本可能需要调整 `compass/src/load_llama.py` 中的返回值。
