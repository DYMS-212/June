# src/utils/progress.py
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import multiprocessing as mp
from queue import Queue, Empty
from tqdm import tqdm
import os

from .logger import get_logger, log_simulation_progress


@dataclass
class ProgressInfo:
    """进度信息数据类"""
    task_id: str
    total: int
    current: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    description: str = ""
    status: str = "running"  # running, completed, failed, paused
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress(self) -> float:
        """进度百分比 (0-1)"""
        return self.current / self.total if self.total > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        """已用时间（秒）"""
        return time.time() - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        """预计剩余时间（秒）"""
        if self.current == 0:
            return None
        rate = self.current / self.elapsed_time
        if rate == 0:
            return None
        remaining = self.total - self.current
        return remaining / rate
    
    @property
    def rate(self) -> float:
        """处理速率（个/秒）"""
        elapsed = self.elapsed_time
        return self.current / elapsed if elapsed > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'total': self.total,
            'current': self.current,
            'progress': self.progress,
            'elapsed_time': self.elapsed_time,
            'eta': self.eta,
            'rate': self.rate,
            'description': self.description,
            'status': self.status,
            'metadata': self.metadata
        }


class ProgressTracker:
    """单个任务的进度跟踪器"""
    
    def __init__(self, 
                 task_id: str,
                 total: int,
                 description: str = "",
                 show_progress_bar: bool = True,
                 log_interval: int = 10,  # 每N%记录一次日志
                 update_callback: Optional[Callable] = None):
        
        self.info = ProgressInfo(
            task_id=task_id,
            total=total,
            description=description
        )
        
        self.show_progress_bar = show_progress_bar
        self.log_interval = log_interval
        self.update_callback = update_callback
        self.logger = get_logger(f"progress.{task_id}")
        
        # 初始化进度条
        self.pbar = None
        if self.show_progress_bar:
            self.pbar = tqdm(
                total=total,
                desc=description,
                unit="item",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        # 日志记录控制
        self.last_log_progress = 0
        
        self.logger.info(f"Started tracking task '{task_id}' with {total} items")
    
    def update(self, n: int = 1, **metadata):
        """更新进度"""
        self.info.current = min(self.info.current + n, self.info.total)
        self.info.last_update = time.time()
        self.info.metadata.update(metadata)
        
        # 更新进度条
        if self.pbar:
            self.pbar.update(n)
            
            # 添加额外信息到进度条
            if metadata:
                postfix_str = ", ".join([f"{k}={v}" for k, v in metadata.items()])
                self.pbar.set_postfix_str(postfix_str)
        
        # 记录日志（按百分比间隔）
        current_progress_percent = int(self.info.progress * 100)
        if current_progress_percent >= self.last_log_progress + self.log_interval:
            self.last_log_progress = current_progress_percent
            log_simulation_progress(
                self.info.task_id,
                self.info.progress,
                {
                    'current': self.info.current,
                    'total': self.info.total,
                    'rate': f"{self.info.rate:.2f} items/s",
                    'eta': f"{self.info.eta:.1f}s" if self.info.eta else "unknown",
                    **metadata
                }
            )
        
        # 调用回调函数
        if self.update_callback:
            self.update_callback(self.info)
    
    def set_description(self, description: str):
        """设置描述"""
        self.info.description = description
        if self.pbar:
            self.pbar.set_description(description)
    
    def complete(self):
        """标记任务完成"""
        self.info.status = "completed"
        self.info.current = self.info.total
        
        if self.pbar:
            self.pbar.close()
        
        self.logger.info(
            f"Task '{self.info.task_id}' completed in {self.info.elapsed_time:.2f}s "
            f"(avg rate: {self.info.rate:.2f} items/s)"
        )
    
    def fail(self, error_message: str = ""):
        """标记任务失败"""
        self.info.status = "failed"
        self.info.metadata['error'] = error_message
        
        if self.pbar:
            self.pbar.close()
        
        self.logger.error(f"Task '{self.info.task_id}' failed: {error_message}")
    
    def pause(self):
        """暂停任务"""
        self.info.status = "paused"
        if self.pbar:
            self.pbar.clear()
    
    def resume(self):
        """恢复任务"""
        self.info.status = "running"
        if self.pbar:
            self.pbar.refresh()
    
    def get_info(self) -> ProgressInfo:
        """获取进度信息"""
        return self.info


class MultiProgressManager:
    """多任务进度管理器"""
    
    def __init__(self, 
                 show_overall_progress: bool = True,
                 log_interval: int = 10):
        
        self.trackers: Dict[str, ProgressTracker] = {}
        self.show_overall_progress = show_overall_progress
        self.log_interval = log_interval
        self.logger = get_logger("progress.manager")
        
        # 总体进度信息
        self.overall_total = 0
        self.overall_current = 0
        self.overall_pbar = None
        
        # 线程安全
        self.lock = threading.Lock()
    
    def create_tracker(self, 
                      task_id: str,
                      total: int,
                      description: str = "",
                      show_progress_bar: bool = True) -> ProgressTracker:
        """创建新的进度跟踪器"""
        
        with self.lock:
            if task_id in self.trackers:
                raise ValueError(f"Task '{task_id}' already exists")
            
            # 更新总体进度
            self.overall_total += total
            
            # 创建跟踪器
            tracker = ProgressTracker(
                task_id=task_id,
                total=total,
                description=description,
                show_progress_bar=show_progress_bar and not self.show_overall_progress,
                update_callback=self._on_tracker_update
            )
            
            self.trackers[task_id] = tracker
            
            # 创建总体进度条（如果需要）
            if self.show_overall_progress and self.overall_pbar is None:
                self.overall_pbar = tqdm(
                    total=self.overall_total,
                    desc="Overall Progress",
                    unit="item",
                    position=0,
                    leave=True
                )
            elif self.show_overall_progress:
                self.overall_pbar.total = self.overall_total
                self.overall_pbar.refresh()
        
        return tracker
    
    def _on_tracker_update(self, progress_info: ProgressInfo):
        """处理跟踪器更新"""
        if self.show_overall_progress and self.overall_pbar:
            # 计算总体进度
            total_current = sum(tracker.info.current for tracker in self.trackers.values())
            delta = total_current - self.overall_current
            self.overall_current = total_current
            
            if delta > 0:
                self.overall_pbar.update(delta)
                
                # 显示当前活动的任务
                active_tasks = [
                    t.info.task_id for t in self.trackers.values() 
                    if t.info.status == "running" and t.info.current < t.info.total
                ]
                if active_tasks:
                    self.overall_pbar.set_postfix_str(f"Active: {', '.join(active_tasks[:3])}")
    
    def get_tracker(self, task_id: str) -> Optional[ProgressTracker]:
        """获取指定的跟踪器"""
        return self.trackers.get(task_id)
    
    def complete_tracker(self, task_id: str):
        """完成指定的跟踪器"""
        if task_id in self.trackers:
            self.trackers[task_id].complete()
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """获取总体进度信息"""
        total_items = sum(t.info.total for t in self.trackers.values())
        completed_items = sum(t.info.current for t in self.trackers.values())
        
        return {
            'total_tasks': len(self.trackers),
            'total_items': total_items,
            'completed_items': completed_items,
            'overall_progress': completed_items / total_items if total_items > 0 else 0,
            'active_tasks': [
                t.info.task_id for t in self.trackers.values() 
                if t.info.status == "running"
            ],
            'completed_tasks': [
                t.info.task_id for t in self.trackers.values() 
                if t.info.status == "completed"
            ],
            'failed_tasks': [
                t.info.task_id for t in self.trackers.values() 
                if t.info.status == "failed"
            ]
        }
    
    def close_all(self):
        """关闭所有进度条"""
        for tracker in self.trackers.values():
            if tracker.pbar:
                tracker.pbar.close()
        
        if self.overall_pbar:
            self.overall_pbar.close()


class MultiProcessProgressManager:
    """多进程进度管理器"""
    
    def __init__(self, manager: mp.Manager):
        self.manager = manager
        self.progress_queue = manager.Queue()
        self.progress_data = manager.dict()
        self.logger = get_logger("progress.multiprocess")
        
        # 启动进度监听线程
        self.monitoring_thread = threading.Thread(target=self._monitor_progress)
        self.monitoring_thread.daemon = True
        self.stop_monitoring = threading.Event()
        self.monitoring_thread.start()
    
    def _monitor_progress(self):
        """监听进度更新"""
        while not self.stop_monitoring.is_set():
            try:
                # 从队列中获取进度更新
                update = self.progress_queue.get(timeout=1.0)
                
                task_id = update['task_id']
                action = update['action']
                
                if action == 'update':
                    self.progress_data[task_id] = update['data']
                    
                    # 记录日志
                    progress = update['data']['progress']
                    if int(progress * 100) % 10 == 0:  # 每10%记录一次
                        log_simulation_progress(
                            task_id,
                            progress,
                            update['data']
                        )
                
                elif action == 'complete':
                    if task_id in self.progress_data:
                        self.progress_data[task_id]['status'] = 'completed'
                    self.logger.info(f"Process task '{task_id}' completed")
                
                elif action == 'fail':
                    if task_id in self.progress_data:
                        self.progress_data[task_id]['status'] = 'failed'
                    self.logger.error(f"Process task '{task_id}' failed: {update.get('error', '')}")
                    
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in progress monitoring: {e}")
    
    def create_process_tracker(self, task_id: str, total: int):
        """为子进程创建进度跟踪器"""
        return ProcessProgressTracker(task_id, total, self.progress_queue)
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """获取所有进度信息"""
        return dict(self.progress_data)
    
    def stop(self):
        """停止监听"""
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)


class ProcessProgressTracker:
    """子进程中使用的进度跟踪器"""
    
    def __init__(self, task_id: str, total: int, queue: mp.Queue):
        self.task_id = task_id
        self.total = total
        self.current = 0
        self.queue = queue
        self.start_time = time.time()
    
    def update(self, n: int = 1, **metadata):
        """更新进度"""
        self.current = min(self.current + n, self.total)
        
        progress_data = {
            'current': self.current,
            'total': self.total,
            'progress': self.current / self.total,
            'elapsed_time': time.time() - self.start_time,
            'process_id': os.getpid(),
            **metadata
        }
        
        self.queue.put({
            'task_id': self.task_id,
            'action': 'update',
            'data': progress_data
        })
    
    def complete(self):
        """标记完成"""
        self.queue.put({
            'task_id': self.task_id,
            'action': 'complete',
            'data': {'final_count': self.current}
        })
    
    def fail(self, error_message: str = ""):
        """标记失败"""
        self.queue.put({
            'task_id': self.task_id,
            'action': 'fail',
            'error': error_message
        })


# 全局进度管理器
_global_progress_manager: Optional[MultiProgressManager] = None


def setup_global_progress_manager(show_overall: bool = True) -> MultiProgressManager:
    """设置全局进度管理器"""
    global _global_progress_manager
    _global_progress_manager = MultiProgressManager(show_overall_progress=show_overall)
    return _global_progress_manager


def get_progress_manager() -> MultiProgressManager:
    """获取全局进度管理器"""
    if _global_progress_manager is None:
        setup_global_progress_manager()
    return _global_progress_manager


def create_progress_tracker(task_id: str, total: int, description: str = "") -> ProgressTracker:
    """便捷函数：创建进度跟踪器"""
    manager = get_progress_manager()
    return manager.create_tracker(task_id, total, description)


if __name__ == "__main__":
    # 测试代码
    import random
    
    # 设置日志
    from .logger import setup_global_logger
    setup_global_logger({
        'name': 'test_progress',
        'format_type': 'human',
        'level': 'INFO'
    })
    
    # 创建进度管理器
    manager = setup_global_progress_manager(show_overall=True)
    
    # 创建多个任务
    tracker1 = manager.create_tracker("circuit_generation", 100, "Generating circuits")
    tracker2 = manager.create_tracker("vqe_simulation", 50, "Running VQE")
    
    # 模拟任务1
    for i in range(100):
        time.sleep(0.1)
        tracker1.update(1, energy=random.uniform(-2, 0))
    tracker1.complete()
    
    # 模拟任务2
    for i in range(50):
        time.sleep(0.2)
        tracker2.update(1, iteration=i+1)
    tracker2.complete()
    
    # 关闭所有进度条
    manager.close_all()
    
    print("进度条测试完成")
