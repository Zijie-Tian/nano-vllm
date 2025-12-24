"""
Chunked Prefill + KV Cache Offload Simulation v2

改进：
1. 简化日志输出
2. 添加reduce时间
3. 计算必须等待KV load完成
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future

# ============== 配置参数 ==============
NUM_CHUNKS = 8
GPU_SLOTS = 4

# 模拟时间 (秒)
TIME_COMPUTE_BLOCK = 0.10    # 计算一个attention block
TIME_REDUCE = 0.03           # 两个partial result做一次reduce
TIME_TRANSFER = 0.08         # 传输一个KV cache
TIME_PROJ = 0.02             # projection生成KV

# ============== 全局时间基准 ==============
START_TIME = None

def now() -> float:
    """返回相对开始的时间(ms)"""
    return (time.time() - START_TIME) * 1000

def log_compute(msg: str):
    """计算队列日志（无缩进）"""
    print(f"[{now():7.1f}ms] [COMPUTE] {msg}")

def log_transfer(msg: str):
    """传输队列日志（缩进）"""
    print(f"[{now():7.1f}ms]          [TRANSFER] {msg}")

def log_info(msg: str):
    """一般信息"""
    print(f"[{now():7.1f}ms] {msg}")

# ============== GPU Slot管理 ==============
class GPUSlots:
    def __init__(self, num_slots: int):
        self.slots = [None] * num_slots  # slot_id -> kv_idx
        self.kv_to_slot = {}             # kv_idx -> slot_id
        self.lock = threading.Lock()
        # KV ready events: kv_idx -> Event
        self.kv_ready = {}
    
    def alloc(self, kv_idx: int) -> int:
        with self.lock:
            for sid, val in enumerate(self.slots):
                if val is None:
                    self.slots[sid] = kv_idx
                    self.kv_to_slot[kv_idx] = sid
                    # 创建ready event
                    if kv_idx not in self.kv_ready:
                        self.kv_ready[kv_idx] = threading.Event()
                    return sid
        raise RuntimeError(f"No free slot for KV{kv_idx}")
    
    def free(self, slot_id: int):
        with self.lock:
            kv_idx = self.slots[slot_id]
            if kv_idx is not None:
                del self.kv_to_slot[kv_idx]
                # 清除event
                if kv_idx in self.kv_ready:
                    del self.kv_ready[kv_idx]
            self.slots[slot_id] = None
    
    def free_kv(self, kv_idx: int):
        with self.lock:
            if kv_idx in self.kv_to_slot:
                sid = self.kv_to_slot[kv_idx]
                self.slots[sid] = None
                del self.kv_to_slot[kv_idx]
                if kv_idx in self.kv_ready:
                    del self.kv_ready[kv_idx]
    
    def mark_ready(self, kv_idx: int):
        """标记KV已就绪（load完成或proj完成）"""
        with self.lock:
            if kv_idx in self.kv_ready:
                self.kv_ready[kv_idx].set()
    
    def wait_ready(self, kv_idx: int):
        """等待KV就绪"""
        with self.lock:
            event = self.kv_ready.get(kv_idx)
        if event:
            event.wait()
    
    def has_kv(self, kv_idx: int) -> bool:
        with self.lock:
            return kv_idx in self.kv_to_slot
    
    def state(self) -> str:
        with self.lock:
            return "[" + "][".join(
                f"KV{v}" if v is not None else "----" 
                for v in self.slots
            ) + "]"

# ============== 操作执行 ==============
class Executor:
    def __init__(self, gpu: GPUSlots):
        self.gpu = gpu
        self.compute_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Compute")
        self.transfer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Transfer")
    
    def proj_kv(self, q_idx: int) -> Future:
        """Projection生成KV，返回Future"""
        def task():
            log_compute(f"PROJ Q{q_idx}->KV{q_idx} START")
            time.sleep(TIME_PROJ)
            slot_id = self.gpu.alloc(q_idx)
            self.gpu.mark_ready(q_idx)  # proj完成，KV立即可用
            log_compute(f"PROJ Q{q_idx}->KV{q_idx} END   slot={slot_id} | {self.gpu.state()}")
            return slot_id
        return self.compute_pool.submit(task)
    
    def compute_attn(self, q_idx: int, kv_indices: list) -> Future:
        """计算attention block，会等待所有KV就绪"""
        def task():
            # 等待所有需要的KV就绪
            for kv_idx in kv_indices:
                self.gpu.wait_ready(kv_idx)
            
            kv_str = ",".join(map(str, kv_indices))
            log_compute(f"ATTN Q{q_idx}*KV[{kv_str}] START")
            time.sleep(TIME_COMPUTE_BLOCK * len(kv_indices))
            log_compute(f"ATTN Q{q_idx}*KV[{kv_str}] END")
            return (q_idx, kv_indices)
        return self.compute_pool.submit(task)
    
    def reduce(self, q_idx: int, num_partials: int) -> Future:
        """Online softmax reduce多个partial结果"""
        def task():
            if num_partials <= 1:
                return
            # n个partial需要n-1次两两reduce
            num_reduces = num_partials - 1
            log_compute(f"REDUCE Q{q_idx} ({num_partials} partials) START")
            time.sleep(TIME_REDUCE * num_reduces)
            log_compute(f"REDUCE Q{q_idx} END")
        return self.compute_pool.submit(task)
    
    def load_kv(self, kv_idx: int) -> Future:
        """从CPU load KV到GPU"""
        def task():
            if self.gpu.has_kv(kv_idx):
                log_transfer(f"LOAD KV{kv_idx} SKIP (already on GPU)")
                return kv_idx
            
            slot_id = self.gpu.alloc(kv_idx)
            log_transfer(f"LOAD KV{kv_idx} START -> slot{slot_id}")
            time.sleep(TIME_TRANSFER)
            self.gpu.mark_ready(kv_idx)  # load完成，标记就绪
            log_transfer(f"LOAD KV{kv_idx} END   | {self.gpu.state()}")
            return kv_idx
        return self.transfer_pool.submit(task)
    
    def offload_kv(self, kv_idx: int) -> Future:
        """从GPU offload KV到CPU"""
        def task():
            log_transfer(f"OFFLOAD KV{kv_idx} START")
            time.sleep(TIME_TRANSFER)
            self.gpu.free_kv(kv_idx)
            log_transfer(f"OFFLOAD KV{kv_idx} END | {self.gpu.state()}")
            return kv_idx
        return self.transfer_pool.submit(task)
    
    def shutdown(self):
        self.compute_pool.shutdown(wait=True)
        self.transfer_pool.shutdown(wait=True)

# ============== 调度器 ==============
def schedule_query(exe: Executor, q_idx: int):
    """调度单个Query的处理"""
    print(f"\n{'='*50}")
    log_info(f"===== Query {q_idx} START =====")
    
    hist_kv = list(range(q_idx))  # 历史KV: 0 ~ q_idx-1
    num_partials = 0
    
    # Phase 1: Projection生成当前KV
    proj_fut = exe.proj_kv(q_idx)
    proj_fut.result()  # 等待完成
    
    # Phase 2: 对角块计算 + 同时prefetch历史KV
    # 启动对角块计算
    diag_fut = exe.compute_attn(q_idx, [q_idx])
    num_partials += 1
    
    # 同时prefetch历史KV (最多3个slot可用)
    prefetch_slots = min(len(hist_kv), GPU_SLOTS - 1)
    prefetch_kv = hist_kv[:prefetch_slots]
    prefetch_futs = [exe.load_kv(kv) for kv in prefetch_kv]
    
    # 等待对角块完成
    diag_fut.result()
    
    # Phase 3: Offload当前KV释放slot
    offload_fut = exe.offload_kv(q_idx)
    
    # 等待prefetch完成，然后计算这批历史KV
    for f in prefetch_futs:
        f.result()
    
    if prefetch_kv:
        hist_fut = exe.compute_attn(q_idx, prefetch_kv)
        num_partials += 1
    else:
        hist_fut = None
    
    # 等待offload完成
    offload_fut.result()
    
    # Phase 4: 处理剩余历史KV
    remaining_kv = hist_kv[prefetch_slots:]
    computed_kv = prefetch_kv.copy()
    
    while remaining_kv:
        # 等待上一批计算完成
        if hist_fut:
            hist_fut.result()
        
        # 释放已计算的KV
        for kv in computed_kv:
            exe.gpu.free_kv(kv)
        
        # Load下一批
        batch_size = min(len(remaining_kv), GPU_SLOTS)
        batch_kv = remaining_kv[:batch_size]
        remaining_kv = remaining_kv[batch_size:]
        
        load_futs = [exe.load_kv(kv) for kv in batch_kv]
        for f in load_futs:
            f.result()
        
        # 计算这批
        hist_fut = exe.compute_attn(q_idx, batch_kv)
        num_partials += 1
        computed_kv = batch_kv
    
    # 等待最后一批计算完成
    if hist_fut:
        hist_fut.result()
    
    # 清理GPU
    for kv in computed_kv:
        exe.gpu.free_kv(kv)
    
    # Phase 5: Reduce所有partial results
    reduce_fut = exe.reduce(q_idx, num_partials)
    reduce_fut.result()
    
    log_info(f"===== Query {q_idx} END =====")

def main():
    global START_TIME
    START_TIME = time.time()
    
    print("Chunked Prefill + KV Cache Offload Simulation v2")
    print(f"Config: {NUM_CHUNKS} chunks, {GPU_SLOTS} GPU slots")
    print(f"Time: compute={TIME_COMPUTE_BLOCK}s, transfer={TIME_TRANSFER}s, reduce={TIME_REDUCE}s")
    
    gpu = GPUSlots(GPU_SLOTS)
    exe = Executor(gpu)
    
    try:
        for q_idx in range(NUM_CHUNKS):
            schedule_query(exe, q_idx)
        
        print(f"\n{'='*50}")
        log_info(f"ALL DONE! Total: {now():.1f}ms")
    finally:
        exe.shutdown()

if __name__ == "__main__":
    main()