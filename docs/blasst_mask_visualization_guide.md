# BLASST 注意力掩码可视化指南

本指南详细说明了如何导出并可视化 BLASST 算子的动态剪枝掩码（Attention Mask Map）。

## 1. 导出原始掩码数据

要可视化算子的决策，首先需要让 Policy 在运行过程中将掩码 Dump 到磁盘。

### 步骤 1：开启 Dump 开关
修改 `nanovllm/kvcache/sparse/blasst.py` 文件，将全局变量 `DEBUG_DUMP_BLASST_MASK` 设为 `True`：

```python
# nanovllm/kvcache/sparse/blasst.py
DEBUG_DUMP_BLASST_MASK = True
```

### 步骤 2：运行推理任务
运行任何使用 BLASST 策略的任务（如 Ruler 测试）。算子会自动为每一个 Query Chunk 和 KV 块生成掩码数据。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):$PYTHONPATH \
    python tests/test_ruler.py \
    --model ~/models/GLM-4-9B-Chat-1M \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --enable-offload \
    --sparse-policy BLASST
```

**数据存放位置**：
原始 `.pt` 文件将保存在 `results/chunked_mask/q_chunk_*/layer_*.pt` 目录下。

---

## 2. 生成热力图可视化

导出数据后，使用 `scripts/plot_blasst_mask.py` 脚本将碎片化的掩码拼接并绘制成热力图。

### 运行绘图脚本
```bash
PYTHONPATH=$(pwd):$PYTHONPATH python scripts/plot_blasst_mask.py --workers 8
```

### 参数说明
- `--source`: 原始数据目录（默认：`results/chunked_mask`）。
- `--output`: 图像输出目录（默认：`results/chunked_mask_map`）。
- `--workers`: 并行绘图的进程数（默认：8）。

---

## 3. 图像分析与解读

生成的图像位于 `results/chunked_mask_map/layer_*.jpg`。每张图包含该层所有的注意力头。

### 核心坐标轴
- **纵轴 (Y-axis)**：Query Blocks（每块 128 tokens）。
- **横轴 (X-axis)**：KV Subblocks（每块 64 tokens）。

### 颜色含义
- **黄色 (1)**：该块参与了计算。
- **深蓝色 (0)**：该块被跳过（剪枝）。

### 关键检查点
1. **Causal 阶梯 (右侧边缘)**：在对角线区域应呈现清晰的阶梯状遮罩，表示因果逻辑正确执行。
2. **LSE 传播模式**：在确立高注意力基准后，右侧应出现大面积的深蓝色区域，表示 LSE 成功抑制了低贡献块。
3. **全局一致性**：热力图应纹理平滑，无明显的 4096 tokens 断层。

---

## 4. 注意事项
- **磁盘空间**：在大上下文（如 128k）下开启 Dump 会产生大量 `.pt` 文件，请及时清理 `results/chunked_mask`。
- **性能影响**：开启 `DEBUG_DUMP_BLASST_MASK` 会因为频繁的磁盘 IO 导致推理速度大幅下降，**生产环境务必保持其为 `False`**。
