# src/simulators/base_simulator.py
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.logger import get_logger, log_simulation_start, log_simulation_complete, log_error
from ..utils.progress import ProgressTracker, ProcessProgressTracker
from ..utils.storage import StorageManager


@dataclass
class SimulationConfig:
    """仿真配置基类"""
    name: str
    enabled: bool = True
    batch_size: int = 100
    max_workers: Optional[int] = None
    n_repeat: int = 1
    save_intermediate: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'enabled': self.enabled,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'n_repeat': self.n_repeat,
            'save_intermediate': self.save_intermediate,
            'metadata': self.metadata
        }


@dataclass
class SimulationResult:
    """仿真结果基类"""
    global_index: int
    batch_index: int
    batch_inner_index: int
    status: str  # success, error, timeout
    time_taken: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'global_index': self.global_index,
            'batch_index': self.batch_index,
            'batch_inner_index': self.batch_inner_index,
            'status': self.status,
            'time_taken': self.time_taken,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class BaseSimulator(ABC):
    """仿真器基类"""
    
    def __init__(self, 
                 config: SimulationConfig,
                 storage_manager: StorageManager,
                 hamiltonian: Any = None,
                 **kwargs):
        
        self.config = config
        self.storage_manager = storage_manager
        self.hamiltonian = hamiltonian
        
        # 设置日志器
        self.logger = get_logger(f"simulator.{config.name}")
        
        # 仿真状态
        self.is_running = False
        self.total_processed = 0
        self.total_success = 0
        self.total_failed = 0
        
        # 结果存储
        self.batch_results: List[List[SimulationResult]] = []
        
        # 初始化仿真器特定设置
        self.setup(**kwargs)
    
    @abstractmethod
    def setup(self, **kwargs):
        """设置仿真器特定参数"""
        pass
    
    @abstractmethod
    def simulate_single(self, 
                       circuit: Any, 
                       global_idx: int, 
                       batch_idx: int, 
                       batch_inner_idx: int,
                       **kwargs) -> SimulationResult:
        """仿真单个电路"""
        pass
    
    def simulate_batch(self, 
                      circuits: List[Any], 
                      batch_id: int,
                      progress_tracker: Optional[ProgressTracker] = None) -> List[SimulationResult]:
        """仿真一个批次的电路"""
        
        batch_start_time = time.time()
        results = []
        
        self.logger.info(f"Starting batch {batch_id} with {len(circuits)} circuits")
        
        if self.config.max_workers == 1 or len(circuits) == 1:
            # 单进程处理
            results = self._simulate_batch_sequential(circuits, batch_id, progress_tracker)
        else:
            # 多进程处理
            results = self._simulate_batch_parallel(circuits, batch_id, progress_tracker)
        
        # 统计结果
        success_count = sum(1 for r in results if r.status == "success")
        failed_count = len(results) - success_count
        
        self.total_processed += len(results)
        self.total_success += success_count
        self.total_failed += failed_count
        
        batch_time = time.time() - batch_start_time
        
        self.logger.info(
            f"Batch {batch_id} completed: {success_count} success, {failed_count} failed, "
            f"time: {batch_time:.2f}s"
        )
        
        # 保存中间结果
        if self.config.save_intermediate:
            self._save_batch_results(results, batch_id)
        
        return results
    
    def _simulate_batch_sequential(self, 
                                  circuits: List[Any], 
                                  batch_id: int,
                                  progress_tracker: Optional[ProgressTracker] = None) -> List[SimulationResult]:
        """顺序处理批次"""
        results = []
        
        for idx, circuit in enumerate(circuits):
            global_idx = self.total_processed + idx
            
            # 重复执行
            best_result = None
            for repeat in range(self.config.n_repeat):
                try:
                    result = self.simulate_single(circuit, global_idx, batch_id, idx)
                    
                    # 选择最好的结果（子类可以重写这个逻辑）
                    if best_result is None or self._is_better_result(result, best_result):
                        best_result = result
                        
                except Exception as e:
                    result = SimulationResult(
                        global_index=global_idx,
                        batch_index=batch_id,
                        batch_inner_index=idx,
                        status="error",
                        time_taken=0.0,
                        error_message=str(e)
                    )
                    if best_result is None:
                        best_result = result
            
            results.append(best_result)
            
            # 更新进度
            if progress_tracker:
                progress_tracker.update(1, 
                                      status=best_result.status,
                                      success_rate=f"{self.total_success/(self.total_processed+1):.2%}")
        
        return results
    
    def _simulate_batch_parallel(self, 
                                circuits: List[Any], 
                                batch_id: int,
                                progress_tracker: Optional[ProgressTracker] = None) -> List[SimulationResult]:
        """并行处理批次"""
        results = [None] * len(circuits)
        max_workers = self.config.max_workers or mp.cpu_count()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_idx = {}
            for idx, circuit in enumerate(circuits):
                global_idx = self.total_processed + idx
                future = executor.submit(
                    self._process_circuit_parallel,
                    circuit, global_idx, batch_id, idx, self.config.n_repeat
                )
                future_to_idx[future] = idx
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    # 更新进度
                    if progress_tracker:
                        progress_tracker.update(1,
                                              status=result.status,
                                              success_rate=f"{self.total_success/(self.total_processed+1):.2%}")
                        
                except Exception as e:
                    self.logger.error(f"Parallel processing error for circuit {idx}: {e}")
                    results[idx] = SimulationResult(
                        global_index=self.total_processed + idx,
                        batch_index=batch_id,
                        batch_inner_index=idx,
                        status="error",
                        time_taken=0.0,
                        error_message=str(e)
                    )
        
        return [r for r in results if r is not None]
    
    def _process_circuit_parallel(self, 
                                 circuit: Any, 
                                 global_idx: int, 
                                 batch_idx: int, 
                                 batch_inner_idx: int,
                                 n_repeat: int) -> SimulationResult:
        """并行处理单个电路（在子进程中执行）"""
        best_result = None
        
        for repeat in range(n_repeat):
            try:
                result = self.simulate_single(circuit, global_idx, batch_idx, batch_inner_idx)
                
                if best_result is None or self._is_better_result(result, best_result):
                    best_result = result
                    
            except Exception as e:
                result = SimulationResult(
                    global_index=global_idx,
                    batch_index=batch_idx,
                    batch_inner_index=batch_inner_idx,
                    status="error",
                    time_taken=0.0,
                    error_message=str(e)
                )
                if best_result is None:
                    best_result = result
        
        return best_result
    
    def _is_better_result(self, new_result: SimulationResult, current_best: SimulationResult) -> bool:
        """判断新结果是否比当前最好结果更好（子类可重写）"""
        # 默认：成功的结果比失败的好
        if new_result.status == "success" and current_best.status != "success":
            return True
        elif new_result.status != "success" and current_best.status == "success":
            return False
        else:
            # 两个都成功或都失败，比较时间（更快的更好）
            return new_result.time_taken < current_best.time_taken
    
    def _save_batch_results(self, results: List[SimulationResult], batch_id: int):
        """保存批次结果"""
        try:
            batch_data = {
                "results": [r.to_dict() for r in results],
                "batch_info": {
                    "batch_id": batch_id,
                    "simulation_type": self.config.name,
                    "total_circuits": len(results),
                    "success_count": sum(1 for r in results if r.status == "success"),
                    "failed_count": sum(1 for r in results if r.status == "error"),
                    "config": self.config.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            filename = f"{self.config.name}_batch_{batch_id}_results"
            self.storage_manager.save_results(
                batch_data, 
                filename, 
                format_type="json",
                description=f"{self.config.name} simulation batch {batch_id} results"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save batch {batch_id} results: {e}")
    
    def run_simulation(self, 
                      circuits: List[Any],
                      progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
        """运行完整的仿真"""
        
        if not self.config.enabled:
            self.logger.info(f"Simulation {self.config.name} is disabled")
            return {"status": "disabled"}
        
        self.is_running = True
        simulation_start_time = time.time()
        
        # 记录仿真开始
        log_simulation_start(self.config.name, self.config.to_dict())
        
        try:
            total_circuits = len(circuits)
            total_batches = (total_circuits + self.config.batch_size - 1) // self.config.batch_size
            
            self.logger.info(f"Starting {self.config.name} simulation: "
                           f"{total_circuits} circuits, {total_batches} batches")
            
            # 批次处理
            for batch_id in range(total_batches):
                start_idx = batch_id * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, total_circuits)
                batch_circuits = circuits[start_idx:end_idx]
                
                batch_results = self.simulate_batch(batch_circuits, batch_id, progress_tracker)
                self.batch_results.append(batch_results)
            
            # 汇总结果
            all_results = []
            for batch in self.batch_results:
                all_results.extend(batch)
            
            simulation_time = time.time() - simulation_start_time
            
            # 创建最终结果
            final_results = {
                "simulation_type": self.config.name,
                "total_circuits": total_circuits,
                "total_processed": self.total_processed,
                "success_count": self.total_success,
                "failed_count": self.total_failed,
                "success_rate": self.total_success / self.total_processed if self.total_processed > 0 else 0,
                "simulation_time": simulation_time,
                "config": self.config.to_dict(),
                "results": [r.to_dict() for r in all_results],
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存最终结果
            self._save_final_results(final_results)
            
            # 记录仿真完成
            log_simulation_complete(self.config.name, {
                "total_circuits": total_circuits,
                "success_count": self.total_success,
                "failed_count": self.total_failed,
                "simulation_time": simulation_time
            })
            
            return final_results
            
        except Exception as e:
            log_error(f"Simulation {self.config.name} failed", e)
            raise
        finally:
            self.is_running = False
    
    def _save_final_results(self, results: Dict[str, Any]):
        """保存最终结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.name}_final_results_{timestamp}"
            
            self.storage_manager.save_results(
                results,
                filename,
                format_type="json",
                description=f"Final {self.config.name} simulation results"
            )
            
            self.logger.info(f"Final results saved as {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取仿真统计信息"""
        return {
            "simulation_type": self.config.name,
            "is_running": self.is_running,
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "total_failed": self.total_failed,
            "success_rate": self.total_success / self.total_processed if self.total_processed > 0 else 0,
            "batches_completed": len(self.batch_results)
        }
    
    def reset(self):
        """重置仿真器状态"""
        self.is_running = False
        self.total_processed = 0
        self.total_success = 0
        self.total_failed = 0
        self.batch_results.clear()
        self.logger.info(f"Simulator {self.config.name} reset")


# 辅助函数，用于检查和验证仿真器
def validate_simulator_config(config: SimulationConfig) -> bool:
    """验证仿真器配置"""
    if not isinstance(config.name, str) or not config.name:
        return False
    
    if config.batch_size <= 0:
        return False
    
    if config.n_repeat <= 0:
        return False
    
    if config.max_workers is not None and config.max_workers <= 0:
        return False
    
    return True


def create_simulator_from_config(simulator_type: str, 
                                config: Dict[str, Any],
                                storage_manager: StorageManager,
                                **kwargs) -> BaseSimulator:
    """根据配置创建仿真器实例"""
    
    # 导入仿真器类
    if simulator_type == "vqe":
        from .vqe_simulator import VQESimulator
        sim_config = VQESimulationConfig(**config)
        return VQESimulator(sim_config, storage_manager, **kwargs)
    
    elif simulator_type == "expressibility":
        from .expressibility_simulator import ExpressibilitySimulator
        sim_config = ExpressibilityConfig(**config)
        return ExpressibilitySimulator(sim_config, storage_manager, **kwargs)
    
    elif simulator_type == "noisy_vqe":
        from .noisy_vqe_simulator import NoisyVQESimulator
        sim_config = NoisyVQEConfig(**config)
        return NoisyVQESimulator(sim_config, storage_manager, **kwargs)
    
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")


# 专用配置类将在各自的仿真器模块中定义
@dataclass
class VQESimulationConfig(SimulationConfig):
    """VQE仿真配置"""
    optimizer: str = "L_BFGS_B"
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6


@dataclass  
class ExpressibilityConfig(SimulationConfig):
    """表达性计算配置"""
    samples: int = 5000
    bins: int = 75


@dataclass
class NoisyVQEConfig(SimulationConfig):
    """含噪声VQE配置"""
    noise_model_config: Dict[str, Any] = field(default_factory=dict)
    shot_count: int = 1024


if __name__ == "__main__":
    # 测试基础仿真器框架
    from ..utils.logger import setup_global_logger
    from ..utils.storage import StorageManager
    
    # 设置日志和存储
    setup_global_logger({'name': 'test_simulator', 'level': 'INFO'})
    storage = StorageManager(base_dir="test_simulator_data")
    
    # 创建测试配置
    config = SimulationConfig(
        name="test_simulation",
        batch_size=10,
        n_repeat=2
    )
    
    # 验证配置
    is_valid = validate_simulator_config(config)
    print(f"配置验证: {'通过' if is_valid else '失败'}")
    
    print("仿真器框架测试完成")
