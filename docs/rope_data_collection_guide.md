# RoPE 数据收集流程

收集各模型在不同 context length 下的 pre/post RoPE QKV tensor 数据，用于 RoPE 缩放分析。

## 概述

通过 `SAVE_ROPE_DIR` 环境变量触发数据保存，在模型 attention 层的 `forward()` 中捕获 RoPE 应用前后的 Q、K 向量以及 V 和 positions。

### 支持的模型

| 模型 | 文件 | 层数 | num_heads | num_kv_heads | head_dim |
|------|------|------|-----------|--------------|----------|
| Qwen2.5-7B-Instruct-1M | `nanovllm/models/qwen2.py` | 28 | 28 | 4 | 128 |
| GLM-4-9B-Chat-1M | `nanovllm/models/glm4.py` | 40 | 32 | 2 | 128 |
| Llama-3.1-8B-Instruct | `nanovllm/models/llama.py` | 32 | 32 | 8 | 128 |

### 两种模式

| 模式 | 条件 | 文件格式 | 适用场景 |
|------|------|---------|----------|
| GPU-only | 短 context (≤64k) | `layer_NN.pt` | 无需 CPU offload |
| Chunked prefill | 长 context (≥128k) | `layer_NN_chunk_MMMM.pt` | 需要 CPU offload |

## 保存的数据格式

每个 `.pt` 文件包含以下字段：

```python
{
    'pre_rope_q': tensor,    # [seq_len, num_heads, head_dim]
    'pre_rope_k': tensor,    # [seq_len, num_kv_heads, head_dim]
    'post_rope_q': tensor,   # [seq_len, num_heads, head_dim]
    'post_rope_k': tensor,   # [seq_len, num_kv_heads, head_dim]
    'v': tensor,             # [seq_len, num_kv_heads, head_dim]
    'positions': tensor,     # [seq_len]
}
```

## 完整流程

### Step 1: 数据收集

使用 `test_ruler.py` + `SAVE_ROPE_DIR` 环境变量运行推理：

**GPU-only 模式 (≤64k)**：
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    SAVE_ROPE_DIR=/home/zijie/Code/nano-vllm/results/kvcache-rope/<model>/<ctx> \
    python tests/test_ruler.py \
    --model ~/models/<MODEL_PATH> \
    --data-dir tests/data/ruler_<ctx> \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len <LEN> \
    --dtype bfloat16
```

**Chunked prefill 模式 (≥128k)**：
```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH \
    SAVE_ROPE_DIR=/home/zijie/Code/nano-vllm/results/kvcache-rope-chunks/<model>/<ctx> \
    python tests/test_ruler.py \
    --model ~/models/<MODEL_PATH> \
    --data-dir tests/data/ruler_<ctx> \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len <LEN> \
    --enable-offload \
    --dtype bfloat16
```

### Step 2: 合并 chunks (仅 chunked 模式)

```bash
python scripts/merge_rope_chunks.py \
    results/kvcache-rope-chunks/<model>/<ctx> \
    results/kvcache-rope/<model>/<ctx> \
    --delete-chunks
```

- `--delete-chunks`: 每层合并后立即删除对应的 chunk 文件，节省磁盘空间（长 context 必须）

### Step 3: 上传到 aliyunpan

```bash
aliyunpan upload results/kvcache-rope/<model>/<ctx> /data/COMPASS/kvcache-rope/<model>/
```

验证上传：
```bash
aliyunpan ls /data/COMPASS/kvcache-rope/<model>/<ctx>/
```

### Step 4: 清理本地文件

```bash
rm -rf results/kvcache-rope/<model>/<ctx>
rm -rf results/kvcache-rope-chunks/<model>/<ctx>  # 如未使用 --delete-chunks
```

## 参数速查

### max-model-len 对应表

| Context | max-model-len |
|---------|---------------|
| 16k | 20000 |
| 32k | 40960 |
| 64k | 72000 |
| 128k | 135000 |
| 256k | 270000 |
| 512k | 540000 |
| 768k | 800000 |

### 数据量估算

每层文件大小 ≈ `seq_len × (num_heads + 3×num_kv_heads + 1) × head_dim × 2 bytes (bf16)`

| 模型 | 128k | 256k | 512k | 768k |
|------|------|------|------|------|
| Qwen2.5-7B (28层) | 59 GB | 119 GB | 238 GB | 357 GB |
| GLM-4-9B (40层) | 76 GB | 152 GB | 285 GB | 570 GB |
| Llama-3.1-8B (32层) | 88 GB | - | - | - |

### chunk 数量计算

`chunks = ceil(seq_len / block_size)`，`block_size = 4096`

| Context | chunks |
|---------|--------|
| 128k | 32 |
| 256k | 64 |
| 512k | 128 |
| 768k | 192 |

总文件数 = `layers × chunks`

## 已收集数据清单

aliyunpan 路径: `/data/COMPASS/kvcache-rope/`

| 模型 | Context Lengths | 每个 ctx 文件数 |
|------|----------------|---------------|
| glm-4-9b | 16k, 32k, 64k, 128k, 256k, 512k, 768k | 40 |
| llama-3.1-8b | 16k, 32k, 64k, 128k | 32 |
| qwen2.5-7b | 16k, 32k, 64k, 128k, 256k, 512k, 768k | 28 |

## 代码位置

| 组件 | 文件 | 说明 |
|------|------|------|
| 保存逻辑 | 已移除（见下方代码模板） | 数据收集完成后移除，需要时重新插入 |
| 合并脚本 | `scripts/merge_rope_chunks.py` | 按层合并 chunk 文件 |

## 保存代码模板

数据收集完成后，保存逻辑已从模型文件中移除。如需重新启用，在各模型的 `Attention.forward()` 中 `self.rotary_emb()` 前后插入以下代码：

```python
        # --- Temporary: save pre-RoPE and post-RoPE QKV ---
        _save_dir = os.environ.get("SAVE_ROPE_DIR", "")
        _should_save = False
        _is_chunked = False
        if _save_dir and self.attn.layer_id >= 0 and q.shape[0] > 128:
            from nanovllm.utils.context import get_context
            _ctx = get_context()
            _is_chunked = _ctx.is_chunked_prefill
            _should_save = True if _is_chunked else not getattr(self, '_rope_saved', False)

        if _should_save:
            pre_rope_q = q.detach().cpu()
            pre_rope_k = k.detach().cpu()
            v_cpu = v.detach().cpu()
            positions_cpu = positions.detach().cpu()

        q, k = self.rotary_emb(positions, q, k)

        if _should_save:
            layer_id = self.attn.layer_id
            if _is_chunked:
                chunk_idx = _ctx.current_chunk_idx
                save_path = os.path.join(_save_dir, f'layer_{layer_id:02d}_chunk_{chunk_idx:04d}.pt')
            else:
                save_path = os.path.join(_save_dir, f'layer_{layer_id:02d}.pt')
                self._rope_saved = True
            torch.save({
                'pre_rope_q': pre_rope_q,
                'pre_rope_k': pre_rope_k,
                'post_rope_q': q.detach().cpu(),
                'post_rope_k': k.detach().cpu(),
                'v': v_cpu,
                'positions': positions_cpu,
            }, save_path)
            print(f"[SAVE_ROPE] Saved layer {layer_id}" +
                  (f" chunk {chunk_idx}" if _is_chunked else "") +
                  f" to {save_path}")
        # --- End temporary save ---
```

### 对正常流程的影响

当 `SAVE_ROPE_DIR` **未设置**时，唯一开销是每次 forward 调用一次 `os.environ.get()`（dict 查找，纳秒级），不影响正常推理。触发条件 `q.shape[0] > 128` 确保 decode 阶段不会保存。

### 插入位置

在各模型的 `forward()` 方法中，替换 `q, k = self.rotary_emb(positions, q, k)` 这一行及其前后。

## 注意事项

1. **磁盘空间**: 768k 数据可达 500GB+，务必使用 `--delete-chunks` 渐进删除
2. **GPU 选择**: 长 context 必须使用 `--enable-offload`，24GB GPU 即可
3. **测试验证**: 数据收集与推理同时进行，RULER 测试结果可验证正确性
4. **上传速度**: aliyunpan 速度波动大 (10-100 MB/s)，大数据量需耐心等待
