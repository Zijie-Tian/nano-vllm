# Task Plan: XAttention BSA 模块化集成

## Goal
将 XAttention BSA 策略按照统一接口集成到 nano-vllm 的 sparse policy 框架中，实现模块化设计。

**最终验证目标**: 运行 `tests/test_ruler.py` 测试 32K 数据的 10 个以内的 sample，得到合理结果（不一定全部 PASS，但结果应在预期精度范围内）。

---

## 强制要求：使用 Hive-Mind 集群思考

**必须使用 Claude Flow MCP 的 hive-mind 集群进行深度推理，提高实现精度。**

### 启动 Hive-Mind 的方式

在每个复杂阶段开始前，必须执行以下步骤：

1. **初始化 Hive-Mind 集群**：
   ```python
   # 通过 MCP 调用
   mcp__claude-flow_alpha__hive-mind_init(
       topology="mesh",  # 或 "hierarchical", "ring", "star"
       maxAgents=5,       # 集群大小
   )
   ```

2. **生成专业代理（Spawning Specialists）**：
   ```python
   # 为不同任务类型创建代理
   mcp__claude-flow_alpha__hive-mind_spawn(
       count=3,
       type="specialist",  # researcher, coder, analyst
   )
   ```

3. **广播思考任务**：
   ```python
   mcp__claude-flow_alpha__hive-mind_broadcast(
       message="分析当前架构设计的潜在问题...",
       priority="high"
   )
   ```

4. **获取集群状态和共识**：
   ```python
   mcp__claude-flow_alpha__hive-mind_status(verbose=True)
   mcp__claude-flow_alpha__hive-mind_consensus(
       action="propose",
       type="design",
       value="模块化接口设计方案"
   )
   ```

### 适用阶段

以下阶段**必须**使用 Hive-Mind 集群思考：

- ✅ Phase 1: SparsePolicy 基类接口确认
- ✅ Phase 2: XAttentionBSAPolicy 接口对齐
- ✅ Phase 3: OffloadEngine 辅助方法模块化
- ✅ Phase 5: attention.py 集成点验证

其他阶段（Phase 4, 6, 7）可以使用标准思考模式。

### 集群配置建议

```yaml
# 推荐配置
topology: mesh          # 网状拓扑，适合并行推理
maxAgents: 5             # 5个专业代理
agentTypes:
  - researcher          # 架构分析
  - coder              # 代码实现
  - analyst            # 接口验证
  - optimizer          # 性能优化
  - validator          # 正确性验证
```

### 输出要求

使用 Hive-Mind 后，必须在计划中记录：
1. 集群产生的关键洞察
2. 多代理共识达成的决策
3. 发现的潜在问题和解决方案

---

## 当前架构分析

### SparsePolicy 基类接口

从 `nanovllm/kvcache/sparse/policy.py` 需要确认基类定义：

```python
class SparsePolicy:
    # 能力标记
    supports_prefill: bool
    supports_decode: bool
    requires_block_selection: bool

    # 核心方法
    def select_blocks(self, available_blocks: List[int], ctx: PolicyContext) -> List[int]

    # 可选方法（prefill 专用）
    def sparse_prefill_attention(self, q, k, v, layer_id) -> torch.Tensor

    # 初始化
    def initialize(self, num_layers, num_kv_heads, head_dim, num_cpu_blocks, dtype, device)
    def reset(self)
```

### 当前 XAttentionBSAPolicy 实现

已实现但需要确认模块化集成的部分：
- `xattn_bsa.py` - 策略类实现
- `config.py` - 枚举和参数
- `sparse/__init__.py` - 策略工厂
- `offload_engine.py` - 辅助方法
- `attention.py` - 集成点

## 详细实现计划

### Phase 1: 确保 SparsePolicy 基类接口统一

**任务**: 验证 `SparsePolicy` 基类定义是否包含所有必需的方法

**步骤**:
1. 读取 `nanovllm/kvcache/sparse/policy.py`
2. 确认基类定义包含：
   - `supports_prefill`, `supports_decode`, `requires_block_selection` 类属性
   - `select_blocks()` 方法
   - `sparse_prefill_attention()` 方法（可选）
   - `initialize()`, `reset()` 方法
3. 如果缺失，补充到基类定义中

**预期结果**: 基类定义完整，所有策略类可以遵循统一接口

---

### Phase 2: XAttentionBSAPolicy 接口对齐

**任务**: 确保 XAttentionBSAPolicy 完全符合 SparsePolicy 接口

**步骤**:
1. 确认 `xattn_bsa.py` 中的类属性正确：
   ```python
   class XAttentionBSAPolicy(SparsePolicy):
       supports_prefill = True
       supports_decode = False
       requires_block_selection = False  # 注意：BSA 内部处理选择
   ```

2. 确保方法签名与基类一致：
   - `select_blocks(available_blocks, ctx) -> List[int]`
   - `sparse_prefill_attention(q, k, v, layer_id) -> Tensor`
   - `initialize(...)`
   - `reset()`

3. 添加文档说明：BSA 在 prefill 阶段内部处理 block 选择，因此 `select_blocks` 返回所有可用块

**预期结果**: XAttentionBSAPolicy 完全符合 SparsePolicy 统一接口

---

### Phase 3: OffloadEngine 辅助方法模块化

**任务**: 确保 OffloadEngine 的辅助方法正确定义且模块化

**步骤**:
1. 确认 `offload_engine.py` 中的辅助方法位置：
   ```python
   # 在 OffloadEngine 类中添加这两个方法
   def load_block_sample_from_cpu(self, cpu_block_id, layer_id, num_samples):
       """加载采样 tokens 用于估算阶段"""
       ...

   def load_block_full_from_cpu(self, cpu_block_id, layer_id):
       """加载完整 block 用于计算阶段"""
       ...
   ```

2. 确保方法签名与 `xattn_bsa.py` 中的调用一致

3. 添加适当的文档说明这两个方法的用途和使用场景

**预期结果**: OffloadEngine 提供统一的 block 加载接口

---

### Phase 4: 模块化集成到工厂模式

**任务**: 确保策略创建通过统一的工厂模式

**步骤**:
1. 检查 `nanovllm/kvcache/__init__.py` 中的 `create_kvcache_manager` 函数

2. 确认策略创建逻辑清晰：
   ```python
   # 根据策略类型构建相应的 kwargs
   if sparse_policy_type == SparsePolicyType.XATTN_BSA:
       policy_kwargs = {
           'block_size': getattr(config, 'sparse_block_size', 128),
           'samples_per_chunk': getattr(config, 'sparse_samples_per_chunk', 128),
           'threshold': getattr(config, 'sparse_threshold', 0.9),
           'use_triton': getattr(config, 'sparse_use_triton', True),
           'stride': getattr(config, sparse_stride', 8),
       }
   ```

3. 确认所有策略类型都有相应的 kwargs 构建逻辑

**预期结果**: 通过 `create_sparse_policy()` 创建所有策略

---

### Phase 5: attention.py 集成点验证

**任务**: 确保 attention.py 中的集成点正确调用策略接口

**步骤**:
1. 检查 `nanovllm/layers/attention.py` 中的 `_chunked_prefill_attention` 方法

2. 确认集成逻辑：
   ```python
   # 检测策略是否有 sparse_prefill_attention 方法
   if sparse_policy is not None and hasattr(sparse_policy, 'sparse_prefill_attention'):
       if sparse_policy.supports_prefill:
           # 使用策略的 sparse_prefill_attention 方法
           o = sparse_policy.sparse_prefill_attention(q, k, v, self.layer_id)
           # 处理异步 offload
           return o

       # 否则使用标准流程（Quest, etc.）
       # ...
   ```

3. 确保没有绕过策略接口直接调用其他逻辑

**预期结果**: attention.py 通过统一的策略接口调用 BSA

---

### Phase 6: 配置参数模块化

**任务**: 确保配置参数结构清晰，易于使用

**步骤**:
1. 检查 `nanovllm/config.py` 中的配置结构

2. 确认 XAttention BSA 参数组织清晰：
   ```python
   # 通用 sparse 参数
   sparse_policy: SparsePolicyType = SparsePolicyType.FULL
   sparse_topk_blocks: int = 8        # Quest
   sparse_threshold_blocks: int = 4     # Quest

   # XATTN_BSA 专用参数
   sparse_block_size: int = 128
   sparse_samples_per_chunk: int = 128
   sparse_threshold: float = 0.9
   sparse_use_triton: bool = True
   sparse_stride: int = 8
   ```

3. 考虑是否需要参数分组或嵌套配置

**预期结果**: 配置参数清晰，易于理解和使用

---

### Phase 7: 模块化验证测试

**任务**: 创建简单的验证脚本确保模块化集成正确

**步骤**:
1. 创建 `tests/test_xattn_bsa_integration.py` 测试脚本

2. 验证以下功能：
   - XAttentionBSAPolicy 可以通过 `create_sparse_policy()` 创建
   - 策略正确响应 `supports_prefill`, `supports_decode` 查询
   - `select_blocks()` 方法返回正确结果
   - OffloadEngine 辅助方法可以正常调用
   - 在模拟环境中策略可以被正确调用

3. 测试用例：
   ```python
   # Test 1: 策略创建
   from nanovllm.config import Config, SparsePolicyType
   from nanovllm.kvcache.sparse import create_sparse_policy

   policy = create_sparse_policy(SparsePolicyType.XATTN_BSA)
   assert hasattr(policy, 'sparse_prefill_attention')
   assert policy.supports_prefill == True
   assert policy.supports_decode == False

   # Test 2: 接口一致性
   # 验证方法签名
   # ...

   # Test 3: OffloadEngine 辅助方法
   # ...
   ```

**预期结果**: 所有测试通过，模块化集成验证成功

---

## 关键设计原则

### 1. 接口统一性
- 所有策略通过 `SparsePolicy` 基类提供统一接口
- 工厂模式创建策略实例
- 策略切换透明，不影响其他模块

### 2. 模块化独立性
- 每个策略类独立实现
- OffloadEngine 提供通用辅助方法
- attention.py 通过策略接口调用，不依赖具体实现

### 3. 可扩展性
- 添加新策略只需：
  1. 创建新的策略类继承 `SparsePolicy`
  2. 添加到 `SparsePolicyType` 枚举
  3. 在工厂函数中添加创建逻辑
  4. 添加相应的配置参数

---

## 文件修改清单

### 必须修改的文件
1. `nanovllm/kvcache/sparse/policy.py` - 确保基类定义完整
2. `nanovllm/kvcache/sparse/xattn_bsa.py` - 确保接口对齐
3. `nanovllm/kvcache/offload_engine.py` - 添加辅助方法
4. `nanovllm/layers/attention.py` - 验证集成点
5. `nanovllm/config.py` - 确认参数结构
6. `nanovllm/kvcache/__init__.py` - 确认工厂模式
7. `nanovllm/kvcache/sparse/__init__.py` - 确认注册逻辑

### 可选创建的文件
- `tests/test_xattn_bsa_integration.py` - 集成验证测试

---

## 实现状态

- [ ] Phase 1: SparsePolicy 基类接口确认
- [ ] Phase 2: XAttentionBSAPolicy 接口对齐
- [ ] Phase 3: OffloadEngine 辅助方法模块化
- [ ] Phase 4: 工厂模式集成验证
- [ ] Phase 5: attention.py 集成点验证
- [ ] Phase 6: 配置参数模块化
- [ ] Phase 7: 模块化验证测试

---

## 备注

- 此计划专注于模块化集成，不涉及算法优化
- 所有修改都遵循现有框架的设计模式
- 重点在于接口统一和模块解耦
- 测试阶段使用简单脚本验证即可，不需要完整的端到端测试
