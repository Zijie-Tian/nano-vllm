# T-MAC QGeMM 迁移与集成指南

## 1. 背景与动机
本项目旨在研究并集成 T-MAC (Table-Lookup-based Mixed-precision ACceleration) 算子。T-MAC 是一种用于低比特大语言模型（LLM）推理的高效算子设计，其核心思想是**通过查表（Lookup Table）代替乘法**，从而加速计算并减少对硬件浮点运算单元的依赖。本指南详细记录了将 T-MAC 基于 TVM 的实现从 `qlutattn` 仓库剥离并作为独立测试组件迁移到 `nano-vllm` 的过程和理解。

## 2. 远期系统架构蓝图：单卡 3090 极致无限上下文异构推理系统
在 `nano-vllm` 中引入 T-MAC 并非为了替换传统的 Linear 层，而是作为**“CPU-GPU异构卸载与动态稀疏注意力”**宏大架构中的关键一环（模块 B：CPU 端的 T-MAC 粗筛与 I/O 收敛）。

整个系统流水线的设计哲学是：**GPU 极致融合生成预测代理 -> CPU T-MAC 极速粗筛与掩码共享 -> PCIe 按需增量传输 -> GPU BLASST 零开销细粒度精算**。

### 四大核心模块概览
1.  **模块 A (GPU 端高度融合的 Epilogue Kernel)**：在 GPU 端生成 KV Cache 时，不仅写出高精度 FP16 用于卸载，同时在 SRAM 内进行 Warp 级池化压缩与 2-bit 量化，并按照 T-MAC 的 Bit-serial 格式打包写出，彻底消除多次访存开销。
2.  **模块 B (CPU 端 T-MAC 粗筛)**：这就是我们目前迁移和验证的核心算子。CPU 接收到 Query 后，利用 AVX 指令极速构建 LUT，对 2-bit 压缩的 KV Cache 进行免乘法 mpGEMM 预测计算，生成全局共享的粗粒度 Mask。
3.  **模块 C (系统级 I/O 调度与增量传输)**：根据 CPU 算出的 Mask，在锁页内存中将散落的 FP16 KV 块压实（Pack），并通过与 GPU Cache 的 Diff 对比，发起单次异步大块 DMA 增量传输。
4.  **模块 D (块稀疏精算 Kernel)**：基于 FlashInfer 架构和 BLASST 算法的 Triton/CUDA Kernel。接收 CPU 传来的 CSR 索引集，执行 Chunked Prefill，并通过 Softmax-Tree 的 LSE 阈值动态剪枝（跳过无用的 Softmax 和 PV 乘加）。

## 3. 当前 T-MAC 算子迁移过程 (Migration Plan)

为了实现模块 B 的基础预研，本次迁移遵循了严格的依赖隔离原则，确保新迁入的 QGeMM TVM 实现作为独立预研组件，不干扰现有的代码架构。

### 3.1 目录结构与隔离
在 `nanovllm/ops/` 目录下创建了专门的子目录：
```text
nanovllm/ops/tvm_qgemm/
├── qgemm.py                 # TVM QGeMM Codegen 入口
├── base.py                  # TVM 编译基础类
├── intrins/                 # TVM 内联 C++ 原语与注册
│   ├── tbl.cc / tbl.py      # 核心查表计算的 TVM Intrinsic
│   ├── lut_ctor.cc / .py    # LUT 预构建的 TVM Intrinsic
│   ├── partial_max.py       # 量化所需的极值提取
│   ├── utils.py             # LLVM IR 生成工具
│   └── __init__.py
└── utils/                   # 隔离的数学与预处理工具
    ├── math_utils.py        
    ├── quant.py             # 纯 NumPy 的量化/反量化模拟
    └── model_utils.py       # (含 preprocess_weights) 权重重排逻辑
```

### 3.2 依赖剥离与导入修正
1.  复制了 `qlutattn` 中的 `QGeMMLUTBitsCodegen` 及 `QGeMMLUTBitsPreprocessorCodegen`。
2.  通过文本处理修改了所有的相对导入，切断了对原库的外部依赖。
3.  迁移了 `quant.py` 和 `model_utils.py` 中的测试与权重预处理逻辑。

### 3.3 测试验证设计
在 `tests/test_tvm_qgemm_align.py` 中构建了独立的对齐测试：
- 使用 2-bit ( M=256, K=128, N=1 ) 测试用例。
- 采用 NumPy 模拟（Reference Implementation）作为 Ground Truth，验证了 TVM 编译算子的正确性。

## 4. 关键问题与解决 (Errors Encountered)
| 错误信息 | 原因分析 | 解决方案 |
|---|---|---|
| `use of undeclared identifier 'SignedHalvingAdder'` | `tbl.cc` 依赖 AVX2 宏，而默认 TVM target 未开启。 | 测试脚本中明确配置 `target = "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2"`。 |

---
*文档更新日期: 2026-03-08*