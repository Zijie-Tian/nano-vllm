---
description: KVCache-RoPE data download rules and verification workflow
---

# KVCache-RoPE Data On-Demand Download

## 数据说明

`results/kvcache-rope/` 存放从阿里云盘下载的 KV Cache + RoPE 数据，用于理论验证分析。

### 云盘路径

```
阿里云盘（备份盘）: /data/COMPASS/kvcache-rope/{model}/{length}/layer_{xx}.pt
```

### 本地路径

```
results/kvcache-rope/{model}/{length}/layer_{xx}.pt
```

### 可用数据

| 模型 | 可用长度 | 层数 | 每层大小 |
|------|----------|------|----------|
| glm-4-9b | 16k, 32k, 64k, 128k | 40 (00-39) | ~603MB |
| llama-3.1-8b | 16k, 32k, 64k, 128k | 40 (00-39) | ~603MB |
| qwen2.5-7b | 16k, 32k | 40 (00-39) | ~603MB |

---

## 强制规则：执行前检查数据

**在执行任何需要 kvcache-rope 数据的代码之前，必须：**

### Step 1: 检查本地文件是否存在

```bash
ls results/kvcache-rope/{model}/{length}/layer_{xx}.pt
```

### Step 2: 如不存在，提示用户下载

提醒用户需要从阿里云盘下载对应数据文件。

下载流程：
1. 创建本地目录：`mkdir -p results/kvcache-rope/{model}/{length}`
2. 清除代理环境变量
3. 设置 savedir、执行下载
4. 移动文件（aliyunpan 会创建嵌套目录 `{savedir}/{uid}/data/COMPASS/...`，需移动到 `{savedir}/layer_{xx}.pt`）
5. 清理嵌套目录

### Step 3: 确认文件就绪后再执行代码

```bash
ls -lh results/kvcache-rope/{model}/{length}/layer_{xx}.pt
# 确认文件大小约 603MB，然后执行代码
```

---

## 批量下载

如果代码需要多层数据，可以一次下载多个：

```bash
# 下载整个目录（所有层）
aliyunpan download /data/COMPASS/kvcache-rope/{model}/{length}/
```

---

## 注意事项

1. **不要跳过检查**：即使上次会话下载过，也要确认文件存在（results/ 在 .gitignore 中）
2. **清除代理**：aliyunpan 使用前必须 `unset` 所有代理环境变量
3. **嵌套路径**：aliyunpan 下载后会在 savedir 下创建 `{uid}/data/COMPASS/...` 嵌套目录，必须移动文件并清理
4. **按需下载**：只下载代码实际需要的层，不要预下载全部 40 层
