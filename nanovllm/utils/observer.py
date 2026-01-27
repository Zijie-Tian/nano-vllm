"""
Observer 基类和 InferenceObserver 实现。

Observer 架构：
- Observer: 基类，定义通用接口
- InferenceObserver: 推理性能观测（TTFT/TPOT）
- MemoryObserver: 内存传输观测（在 memory_observer.py 中定义）
"""


class Observer:
    """
    Observer 基类，提供通用的启用/禁用、重置、输出接口。

    所有 Observer 子类应继承此类并实现：
    - complete_reset(): 重置所有统计数据
    - get_summary(): 返回统计摘要 dict
    - print_summary(): 打印人类可读的摘要
    """

    _enabled: bool = True  # 默认启用

    @classmethod
    def enable(cls) -> None:
        """启用 observer"""
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """禁用 observer"""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """检查是否启用"""
        return cls._enabled

    @classmethod
    def complete_reset(cls) -> None:
        """重置所有统计数据（子类实现）"""
        raise NotImplementedError

    @classmethod
    def get_summary(cls) -> dict:
        """返回统计摘要（子类实现）"""
        raise NotImplementedError

    @classmethod
    def print_summary(cls) -> None:
        """打印人类可读的摘要（子类可选覆盖）"""
        import json
        print(json.dumps(cls.get_summary(), indent=2))


class InferenceObserver(Observer):
    """
    推理性能 Observer，统计 TTFT 和 TPOT。

    - TTFT (Time To First Token): 首个 token 生成延迟
    - TPOT (Time Per Output Token): 每个输出 token 的平均延迟

    统计位置：
    - TTFT 开始: scheduler.py:35-36 - 第一个 sequence 从 waiting 队列取出时
    - TTFT 结束: llm_engine.py:69-72 - prefill 完成后（包括 chunked prefill 所有 chunks）
    - TPOT 开始: llm_engine.py:65 - 每次 decode step 结束时
    - TPOT 结束: llm_engine.py:62-63 - 下一次 decode step 开始时计算（测量上一次 decode 时间）
    - 重置: llm_engine.py:97 - generate() 开始时

    注意：TPOT 需要至少 2 个输出 token 才能计算（测量 decode step 间隔）。
    """

    # 时间戳 (nanoseconds)
    ttft_start: int = 0
    tpot_start: int = 0

    # 统计结果 (nanoseconds)
    ttft: int = 0
    tpot: int = 0

    @classmethod
    def reset_ttft(cls) -> None:
        """重置 TTFT 计时器"""
        cls.ttft_start = 0

    @classmethod
    def complete_reset(cls) -> None:
        """重置所有统计数据"""
        cls.ttft_start = 0
        cls.tpot_start = 0
        cls.ttft = 0
        cls.tpot = 0

    @classmethod
    def get_summary(cls) -> dict:
        """返回统计摘要"""
        return {
            "ttft_ns": cls.ttft,
            "ttft_ms": cls.ttft / 1e6,
            "tpot_ns": cls.tpot,
            "tpot_ms": cls.tpot / 1e6,
        }

    @classmethod
    def print_summary(cls) -> None:
        """打印摘要"""
        print(f"[InferenceObserver] TTFT: {cls.ttft / 1e6:.2f}ms, TPOT: {cls.tpot / 1e6:.2f}ms")
