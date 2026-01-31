"""
DensityObserver - Sparse Attention Density 统计 Observer。

统计两种 density:
1. Compute Density (计算密度): 基于 BSA block size (128)
   - density = selected_bsa_blocks / total_causal_bsa_blocks
   - GPU-only 和 Offload 模式应该一致

2. Communication Density (通信密度): 基于 CPU block size (如 4096)
   - comm_density = selected_cpu_blocks / total_cpu_blocks
   - 仅用于 Offload 模式，由于粒度更粗，必然 >= compute density

统计位置：
- GPU-only: xattn_bsa.py compute_prefill() - 只记录 compute density
- Offload: xattn_bsa.py select_blocks() - 记录两种 density

对于 Offload 模式的 Density 计算:
- 不是简单的 avg 或 min
- 而是 sum(selected) / sum(total)，正确处理不同 chunk 大小的权重
"""

from typing import List, Dict, Optional, Tuple
import torch
from nanovllm.utils.observer import Observer


class DensityObserver(Observer):
    """
    Sparse Attention Density Observer。

    记录每层的 density，用于验证 GPU-only 和 Offload 模式的一致性。

    使用方式:
        DensityObserver.enable()
        DensityObserver.complete_reset()
        # ... run inference ...
        DensityObserver.record(layer_id, mask, causal=True)
        # 或者使用累积模式 (offload):
        DensityObserver.record_counts(layer_id, selected, total)
        # ...
        DensityObserver.print_summary()
    """

    _enabled: bool = False  # 默认禁用

    # 每层的 compute density 记录 (BSA block 粒度)
    # key: layer_id, value: list of density values (每次 prefill chunk 一个)
    _layer_densities: Dict[int, List[float]] = {}

    # 每层的 communication density 记录 (CPU block 粒度，仅 offload 模式)
    _layer_comm_densities: Dict[int, List[float]] = {}

    # 累积模式: 记录 selected/total counts (用于 offload 模式)
    # 这样可以在所有 chunks 完成后正确计算 density = sum(selected) / sum(total)
    _layer_selected_counts: Dict[int, List[int]] = {}
    _layer_total_counts: Dict[int, List[int]] = {}

    # Mask shape 记录 (用于调试)
    _last_q_blocks: int = 0
    _last_k_blocks: int = 0

    # 模式标记
    _mode: str = "unknown"  # "gpu_only" or "offload"

    @classmethod
    def set_mode(cls, mode: str) -> None:
        """设置当前模式 (gpu_only / offload)"""
        cls._mode = mode

    @classmethod
    def record(
        cls,
        layer_id: int,
        mask: torch.Tensor,
        causal: bool = True,
    ) -> float:
        """
        记录一层的 density (适用于 GPU-only 模式)。

        Args:
            layer_id: 层 ID
            mask: [batch, heads, q_blocks, k_blocks] boolean tensor
            causal: 是否考虑 causal mask (只计算下三角)

        Returns:
            density 值
        """
        if not cls._enabled:
            return 0.0

        density = cls._compute_density(mask, causal)

        # 记录
        if layer_id not in cls._layer_densities:
            cls._layer_densities[layer_id] = []
        cls._layer_densities[layer_id].append(density)

        # 记录 mask shape
        cls._last_q_blocks = mask.shape[2]
        cls._last_k_blocks = mask.shape[3]

        return density

    @classmethod
    def record_counts(
        cls,
        layer_id: int,
        selected_blocks: int,
        total_blocks: int,
    ) -> None:
        """
        记录一层的 selected/total block counts (适用于 offload 累积模式)。

        使用累积计数而不是直接计算 density，这样在所有 chunks 处理完后可以正确计算：
        overall_density = sum(selected) / sum(total)

        这比 avg(density) 更准确，因为不同 chunk 的 Q 和 K 长度不同。

        Args:
            layer_id: 层 ID
            selected_blocks: 这个 chunk 选中的 blocks 数量
            total_blocks: 这个 chunk 的 total possible blocks 数量
        """
        if not cls._enabled:
            return

        # 初始化列表
        if layer_id not in cls._layer_selected_counts:
            cls._layer_selected_counts[layer_id] = []
        if layer_id not in cls._layer_total_counts:
            cls._layer_total_counts[layer_id] = []

        # 累积记录
        cls._layer_selected_counts[layer_id].append(selected_blocks)
        cls._layer_total_counts[layer_id].append(total_blocks)

    @classmethod
    def record_comm_density(
        cls,
        layer_id: int,
        selected_cpu_blocks: int,
        total_cpu_blocks: int,
    ) -> float:
        """
        记录一层的 communication density (CPU block 粒度)。

        Args:
            layer_id: 层 ID
            selected_cpu_blocks: 选中的 CPU blocks 数量
            total_cpu_blocks: 总 CPU blocks 数量

        Returns:
            communication density 值
        """
        if not cls._enabled:
            return 0.0

        if total_cpu_blocks == 0:
            return 1.0

        comm_density = selected_cpu_blocks / total_cpu_blocks

        # 记录
        if layer_id not in cls._layer_comm_densities:
            cls._layer_comm_densities[layer_id] = []
        cls._layer_comm_densities[layer_id].append(comm_density)

        return comm_density

    @classmethod
    def _compute_density(cls, mask: torch.Tensor, causal: bool) -> float:
        """计算 mask 的 density"""
        batch, heads, q_blocks, k_blocks = mask.shape

        if causal:
            # 只计算下三角区域
            causal_mask = torch.tril(
                torch.ones(q_blocks, k_blocks, device=mask.device, dtype=torch.bool)
            )
            total_blocks = causal_mask.sum().item() * batch * heads
            selected_blocks = (mask & causal_mask.unsqueeze(0).unsqueeze(0)).sum().item()
        else:
            total_blocks = mask.numel()
            selected_blocks = mask.sum().item()

        if total_blocks == 0:
            return 1.0

        return selected_blocks / total_blocks

    @classmethod
    def complete_reset(cls) -> None:
        """重置所有统计"""
        cls._layer_densities = {}
        cls._layer_comm_densities = {}
        cls._layer_selected_counts = {}
        cls._layer_total_counts = {}
        cls._last_q_blocks = 0
        cls._last_k_blocks = 0
        cls._mode = "unknown"

    @classmethod
    def get_per_layer_density(cls) -> Dict[int, float]:
        """
        获取每层的 density。

        对于累积模式 (offload): density = sum(selected) / sum(total)
        对于直接记录模式 (gpu_only): density = avg(density_values)
        """
        result = {}

        # 优先使用累积模式 (offload)
        if cls._layer_selected_counts:
            for layer_id in cls._layer_selected_counts:
                selected_list = cls._layer_selected_counts.get(layer_id, [])
                total_list = cls._layer_total_counts.get(layer_id, [])
                total_selected = sum(selected_list)
                total_total = sum(total_list)
                if total_total > 0:
                    result[layer_id] = total_selected / total_total
        else:
            # 直接记录模式 (gpu_only)
            for layer_id, densities in cls._layer_densities.items():
                if densities:
                    result[layer_id] = sum(densities) / len(densities)

        return result

    @classmethod
    def get_overall_density(cls) -> float:
        """
        获取所有层的总体 compute density。

        对于累积模式 (offload): density = sum(all_selected) / sum(all_total)
        对于直接记录模式 (gpu_only): density = avg(all_density_values)

        注意: 总体 density 不是简单的 avg(per_layer_density)，
        而是 sum(all_selected) / sum(all_total)，这样可以正确处理权重。
        """
        # 优先使用累积模式 (offload)
        if cls._layer_selected_counts:
            total_selected = 0
            total_total = 0
            for layer_id in cls._layer_selected_counts:
                total_selected += sum(cls._layer_selected_counts[layer_id])
                total_total += sum(cls._layer_total_counts.get(layer_id, []))
            if total_total > 0:
                return total_selected / total_total
            return 0.0

        # 直接记录模式 (gpu_only)
        all_densities = []
        for densities in cls._layer_densities.values():
            all_densities.extend(densities)
        if not all_densities:
            return 0.0
        return sum(all_densities) / len(all_densities)

    @classmethod
    def get_overall_comm_density(cls) -> float:
        """获取所有层的平均 communication density"""
        all_densities = []
        for densities in cls._layer_comm_densities.values():
            all_densities.extend(densities)
        if not all_densities:
            return 0.0
        return sum(all_densities) / len(all_densities)

    @classmethod
    def get_summary(cls) -> dict:
        """返回统计摘要"""
        per_layer = cls.get_per_layer_density()
        return {
            "mode": cls._mode,
            "overall_density": cls.get_overall_density(),
            "per_layer_density": per_layer,
            "num_layers": len(per_layer),
            "last_mask_shape": {
                "q_blocks": cls._last_q_blocks,
                "k_blocks": cls._last_k_blocks,
            },
        }

    @classmethod
    def get_min_density(cls) -> Tuple[int, float]:
        """获取最低 density 的层和值"""
        per_layer = cls.get_per_layer_density()
        if not per_layer:
            return -1, 0.0
        min_layer = min(per_layer, key=per_layer.get)
        return min_layer, per_layer[min_layer]

    @classmethod
    def print_summary(cls) -> None:
        """打印人类可读的摘要"""
        per_layer = cls.get_per_layer_density()
        overall = cls.get_overall_density()
        min_layer, min_density = cls.get_min_density()
        overall_comm = cls.get_overall_comm_density()

        print(f"[DensityObserver] Mode: {cls._mode}")
        print(f"  Compute density: {overall:.4f} (min: {min_density:.4f} @ layer {min_layer})")
        if overall_comm > 0:
            print(f"  Comm density: {overall_comm:.4f}")
        print(f"  Num layers: {len(per_layer)}")
        # 输出 layer 0 的 density 用于对比
        if 0 in per_layer:
            print(f"  Layer 0 density: {per_layer[0]:.6f}")
