"""
MemoryObserver - 内存传输统计 Observer。

统计 GPU-CPU 间的数据传输量：
- H2D (Host to Device): CPU → GPU
- D2H (Device to Host): GPU → CPU
- D2D (Device to Device): GPU → GPU (buffer copy)
"""

from nanovllm.utils.observer import Observer


class MemoryObserver(Observer):
    """
    内存传输 Observer，统计 GPU-CPU 间的数据传输量。

    统计类型：
    - H2D (Host to Device): CPU → GPU
    - D2H (Device to Host): GPU → CPU
    - D2D (Device to Device): GPU → GPU (buffer copy)

    统计位置（均在 offload_engine.py）：
    - H2D: load_to_slot_layer(), load_block_sample_from_cpu(), load_block_full_from_cpu()
    - D2H: offload_slot_layer_to_cpu(), offload_prefill_buffer_async()
    - D2D: write_to_prefill_buffer(), write_to_decode_buffer()
    - 重置: llm_engine.py:generate() - 与 InferenceObserver 一起重置
    """

    _enabled: bool = False  # 默认禁用，需要显式启用

    # H2D 统计
    h2d_bytes: int = 0
    h2d_count: int = 0

    # D2H 统计
    d2h_bytes: int = 0
    d2h_count: int = 0

    # D2D 统计
    d2d_bytes: int = 0
    d2d_count: int = 0

    # 按阶段统计
    prefill_h2d_bytes: int = 0
    prefill_d2h_bytes: int = 0
    decode_h2d_bytes: int = 0
    decode_d2h_bytes: int = 0

    @classmethod
    def record_h2d(cls, num_bytes: int, is_prefill: bool = True) -> None:
        """记录 H2D 传输"""
        if not cls._enabled:
            return
        cls.h2d_bytes += num_bytes
        cls.h2d_count += 1
        if is_prefill:
            cls.prefill_h2d_bytes += num_bytes
        else:
            cls.decode_h2d_bytes += num_bytes

    @classmethod
    def record_d2h(cls, num_bytes: int, is_prefill: bool = True) -> None:
        """记录 D2H 传输"""
        if not cls._enabled:
            return
        cls.d2h_bytes += num_bytes
        cls.d2h_count += 1
        if is_prefill:
            cls.prefill_d2h_bytes += num_bytes
        else:
            cls.decode_d2h_bytes += num_bytes

    @classmethod
    def record_d2d(cls, num_bytes: int) -> None:
        """记录 D2D 传输"""
        if not cls._enabled:
            return
        cls.d2d_bytes += num_bytes
        cls.d2d_count += 1

    @classmethod
    def complete_reset(cls) -> None:
        """重置所有统计"""
        cls.h2d_bytes = cls.h2d_count = 0
        cls.d2h_bytes = cls.d2h_count = 0
        cls.d2d_bytes = cls.d2d_count = 0
        cls.prefill_h2d_bytes = cls.prefill_d2h_bytes = 0
        cls.decode_h2d_bytes = cls.decode_d2h_bytes = 0

    @classmethod
    def get_summary(cls) -> dict:
        """返回统计摘要"""
        return {
            "total": {
                "h2d_bytes": cls.h2d_bytes,
                "h2d_count": cls.h2d_count,
                "d2h_bytes": cls.d2h_bytes,
                "d2h_count": cls.d2h_count,
                "d2d_bytes": cls.d2d_bytes,
                "d2d_count": cls.d2d_count,
            },
            "prefill": {
                "h2d_bytes": cls.prefill_h2d_bytes,
                "d2h_bytes": cls.prefill_d2h_bytes,
            },
            "decode": {
                "h2d_bytes": cls.decode_h2d_bytes,
                "d2h_bytes": cls.decode_d2h_bytes,
            },
        }

    @classmethod
    def _fmt_bytes(cls, b: int) -> str:
        """格式化字节数"""
        if b >= 1e9:
            return f"{b/1e9:.2f} GB"
        if b >= 1e6:
            return f"{b/1e6:.2f} MB"
        if b >= 1e3:
            return f"{b/1e3:.2f} KB"
        return f"{b} B"

    @classmethod
    def print_summary(cls) -> None:
        """打印人类可读的摘要"""
        fmt = cls._fmt_bytes
        total = cls.h2d_bytes + cls.d2h_bytes + cls.d2d_bytes
        print(f"[MemoryObserver] Total: {fmt(total)}")
        print(f"  H2D: {fmt(cls.h2d_bytes)} ({cls.h2d_count} ops)")
        print(f"  D2H: {fmt(cls.d2h_bytes)} ({cls.d2h_count} ops)")
        print(f"  D2D: {fmt(cls.d2d_bytes)} ({cls.d2d_count} ops)")
        print(f"  Prefill - H2D: {fmt(cls.prefill_h2d_bytes)}, D2H: {fmt(cls.prefill_d2h_bytes)}")
        print(f"  Decode  - H2D: {fmt(cls.decode_h2d_bytes)}, D2H: {fmt(cls.decode_d2h_bytes)}")
