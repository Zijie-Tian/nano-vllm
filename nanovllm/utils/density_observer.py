"""
DensityObserver - Sparse Attention Density 统计 Observer。

统计每层的 sparse attention density:
- density = selected_blocks / total_causal_blocks
- 在 causal attention 下，只计算下三角区域

统计位置：
- GPU-only: xattn_bsa.py compute_prefill()
- Offload: xattn_bsa.py select_blocks()
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
        # ...
        DensityObserver.print_summary()
    """

    _enabled: bool = False  # 默认禁用

    # 每层的 density 记录
    # key: layer_id, value: list of density values (每次 prefill chunk 一个)
    _layer_densities: Dict[int, List[float]] = {}

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
        记录一层的 density。

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
        cls._last_q_blocks = 0
        cls._last_k_blocks = 0
        cls._mode = "unknown"

    @classmethod
    def get_per_layer_density(cls) -> Dict[int, float]:
        """获取每层的平均 density"""
        result = {}
        for layer_id, densities in cls._layer_densities.items():
            if densities:
                result[layer_id] = sum(densities) / len(densities)
        return result

    @classmethod
    def get_overall_density(cls) -> float:
        """获取所有层的平均 density"""
        all_densities = []
        for densities in cls._layer_densities.values():
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

        print(f"[DensityObserver] Mode: {cls._mode}")
        print(f"  Overall density: {overall:.4f}")
        print(f"  Min density: {min_density:.4f} (layer {min_layer})")
        print(f"  Num layers: {len(per_layer)}")
