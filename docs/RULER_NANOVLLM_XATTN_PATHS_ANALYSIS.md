# RULER XAttention 双路径分析

**创建时间**: 2026-01-14

---

## 一、两条独立的 XAttention 实现路径

### 路径 A: COMPASS 源码实现 (HuggingFace 后端)

**实现位置**: `compass/src/Xattention.py`

**调用方式**:
```python
# call_api.py 创建 FastPrefillConfig
fastprefillconfig = FastPrefillConfig(
    metric=args.metric,      # "xattn", "compass", "full", "minfer", "avgpool"
    threshold=args.threshold,  # float: 0.9
    stride=args.stride,        # int: 8/16
)

# load_llama.py 根据 metric 选择函数
if self.fastprefillconfig.metric == "xattn":
    attn_output = Xattention_prefill(
        query_states, key_states, value_states,
        stride, norm=1, threshold=threshold
    )
elif self.fastprefillconfig.metric == "compass":
    attn_output = Compass_prefill(
        query_states, key_states, value_states,
        lambd=3.0  # 不同于 xattn 的 threshold
    )
elif self.fastprefillconfig.metric == "full":
    attn_output = Full_prefill(...)
```

**特点**:
- ✅ 已完全实现
- ✅ 支持多种 metric (xattn, compass, full, minfer, avgpool)
- ✅ 参数可通过命令行传递 (`--threshold`, `--stride`)
- ❌ 不支持 CPU offload（长上下文会 OOM）

### 路径 B: Nanovllm 内部实现 (Nanovllm 后端)

**实现位置**: `3rdparty/nanovllm/nanovllm/kvcache/sparse/xattn.py`

**当前状态**:
```python
# model_wrappers.py 硬编码
sparse_policy: SparsePolicyType.FULL,  # ❌ 不使用 metric 参数

# nanovllm 实际支持的策略
SparsePolicyType.FULL        # 完整注意力
SparsePolicyType.XATTN       # XAttention 稀疏注意力
SparsePolicyType.MINFERENCE  # MInference 垂直+slash
SparsePolicyType.QUEST       # Quest decode-only
```

**特点**:
- ✅ 支持 CPU offload（可处理 32k+ 上下文）
- ✅ 已实现 XAttention 策略
- ❌ 当前 RULER 未使用 metric → sparse_policy 映射
- ❌ 参数未暴露给命令行

---

## 二、核心差异分析

### 2.1 架构差异

| 维度 | COMPASS 源码 (HF) | Nanovllm 后端 |
|------|-------------------|---------------|
| 实现位置 | `compass/src/Xattention.py` | `nanovllm/.../xattn.py` |
| 参数名称 | `threshold` (float 0-1) | `xattn_threshold` (float 0-1) |
| 参数含义 | Attention mass 阈值 | Block 选择阈值 |
| 其他参数 | `stride` | `xattn_stride`, `xattn_chunk_size` |
| 执行模式 | 单 GPU，无 offload | CPU offload 支持 |

### 2.2 "数据并行"的含义

**XAttention 中的 "Chunked"**:
```python
# nanovllm XAttention 的 chunked estimation
for chunk_idx in range(q_chunk_num):
    # 分块计算 QK^T
    attn_weights_slice = flat_group_gemm_fuse_reshape(...)
    # 分块 softmax + 聚合
    attn_sum = softmax_fuse_block_sum(...)
```

这是 **算法层面的分块计算**，不是分布式数据并行。

**RULER 测试的并行**:
- 当前已实现：任务级并行（多个 GPU 运行不同任务）
- 未实现：单任务内的数据并行（nanovllm 不支持）

### 2.3 参数语义差异

**COMPASS XAttention 参数**:
```python
# compass/src/Xattention.py
Xattention_prefill(
    query_states, key_states, value_states,
    stride=8,
    norm=1,
    threshold=0.9,  # attention mass 阈值
    use_triton=True
)
```

**Nanovllm XAttention 参数**:
```python
# nanovllm/kvcache/sparse/xattn.py
XAttentionPolicy(
    stride=8,
    threshold=0.9,     # block 选择阈值 (不同语义！)
    chunk_size=16384,  # 额外参数
    use_triton=True,
    keep_sink=False,
    keep_recent=False,
    norm=1.0
)
```

---

## 三、集成方案修正

### 3.1 问题定义

**当前状态**:
1. HF 后端 (`--metric xattn`) → 使用 COMPASS 源码 XAttention ✅
2. Nanovllm 后端 (`--metric xattn`) → 硬编码 FULL，忽略 metric ❌

**目标**:
让 Nanovllm 后端也支持 `--metric xattn`，使用 nanovllm 内部的 XAttention 实现

### 3.2 修正后的集成方案

#### 方案 A: 简单映射（推荐）

```python
# call_api.py get_llm() 函数
elif args.server_type == 'nanovllm':
    from model_wrappers import NanoVLLMModel

    # Metric → SparsePolicy 映射
    metric_to_policy = {
        'full': 'FULL',
        'xattn': 'XATTN',      # 使用 nanovllm 的 XAttention
        'compass': 'XATTN',    # 同上
        'minfer': 'MINFERENCE',
        'avgpool': 'FULL',      # 无对应，使用 FULL
    }

    sparse_policy = metric_to_policy.get(args.metric, 'FULL')

    # 映射 COMPASS 参数到 nanovllm 参数
    # 注意: 语义可能不完全相同，需要调整
    llm = NanoVLLMModel(
        name_or_path=args.model_name_or_path,
        sparse_policy=sparse_policy,
        # Nanovllm XAttention 参数
        xattn_threshold=float(os.environ.get(
            'NANOVLLM_XATTN_THRESHOLD',
            args.threshold if args.metric in ['xattn', 'compass'] else 0.9
        )),
        xattn_stride=int(os.environ.get(
            'NANOVLLM_XATTN_STRIDE',
            args.stride if args.metric in ['xattn', 'compass'] else 8
        )),
        # ... 其他参数
    )
```

#### 方案 B: 分别配置（更灵活）

```python
# 支持独立的 nanovllm 参数
llm = NanoVLLMModel(
    name_or_path=args.model_name_or_path,
    sparse_policy=sparse_policy,
    # 优先使用 nanovllm 专用环境变量
    xattn_threshold=float(os.environ.get(
        'NANOVLLM_XATTN_THRESHOLD',
        # 对于 'compass' metric，使用不同的默认值
        5.0 if args.metric == 'compass' else 0.9
    )),
    xattn_stride=int(os.environ.get(
        'NANOVLLM_XATTN_STRIDE',
        args.stride if hasattr(args, 'stride') else 8
    )),
)
```

### 3.3 关键注意事项

1. **参数不等价**:
   - COMPASS `threshold=0.9` ≠ Nanovllm `xattn_threshold=0.9`
   - COMPASS compass `lambd=5.0` ≠ Nanovllm `xattn_threshold`
   - 需要通过实验找到等效的参数值

2. **独立测试**:
   - `--metric xattn` + HF 后端 = COMPASS XAttention
   - `--metric xattn` + Nanovllm 后端 = Nanovllm XAttention
   - 两者结果可能不同，需要分别验证

3. **文档说明**:
   - 需要在文档中清楚说明两条路径的区别
   - 建议使用不同的命名区分：
     - HF 后端: `--metric xattn-hf`
     - Nanovllm 后端: `--metric xattn-nanovllm` (或自动检测)

---

## 四、测试验证计划

### 4.1 双路径对比测试

```bash
# 测试 1: COMPASS 源码 XAttention (HF 后端)
CUDA_VISIBLE_DEVICES=0 \
bash run.sh llama3.1-8b-chat synthetic --metric xattn --task niah_single_1

# 测试 2: Nanovllm XAttention (Nanovllm 后端)
CUDA_VISIBLE_DEVICES=0 \
bash run.sh llama3.1-8b-nanovllm synthetic --metric xattn --task niah_single_1

# 测试 3: 参数调优测试
export NANOVLLM_XATTN_THRESHOLD=0.95
CUDA_VISIBLE_DEVICES=0 \
bash run.sh llama3.1-8b-nanovllm synthetic --metric xattn --task niah_single_1
```

### 4.2 验证要点

| 验证项 | HF 后端 | Nanovllm 后端 |
|--------|---------|---------------|
| XAttention 生效 | 检查日志中的 "Xattention_prefill" | 检查 "XAttentionPolicy" |
| 参数传递 | FastPrefillConfig.threshold | xattn_threshold |
| 准确率 | 记录基准 | 对比基准 |
| 内存使用 | 可能 OOM (32k+) | CPU offload 支持 |

---

## 五、更新后的集成步骤

### Phase 1: 修改 call_api.py

添加 metric → sparse_policy 映射（见方案 A）

### Phase 2: 修改 model_wrappers.py

添加 sparse_policy 和 xattn_* 参数支持

### Phase 3: 参数调优

通过实验找到 nanovllm XAttention 的最佳参数：
```bash
# 参数搜索
for THRESHOLD in 0.7 0.8 0.9 0.95; do
    export NANOVLLM_XATTN_THRESHOLD=$THRESHOLD
    bash run.sh llama3.1-8b-nanovllm synthetic --metric xattn --task niah_single_1
done
```

### Phase 4: 文档更新

在 CLAUDE.md 中添加双路径说明：
```markdown
## XAttention 测试指南

### HuggingFace 后端 (COMPASS 源码)
使用 `compass/src/Xattention.py` 实现
bash run.sh llama3.1-8b-chat synthetic --metric xattn

### Nanovllm 后端 (Nanovllm 内部实现)
使用 `nanovllm/kvcache/sparse/xattn.py` 实现
bash run.sh llama3.1-8b-nanovllm synthetic --metric xattn

注意: 两条路径使用不同的参数和实现，结果可能不同。
```

---

## 六、开放问题

1. **参数等价性**: COMPASS `threshold=0.9` 对应 nanovllm 的什么值？
2. **性能对比**: 两条路径的速度和准确率对比如何？
3. **命名冲突**: 是否需要区分 `--metric xattn` 的两条路径？
4. **数据并行**: Nanovllm 是否需要/支持真正的数据并行？

---

**文档版本**: 2.0 (修正版)
**最后更新**: 2026-01-14
**状态**: 待确认和实施
