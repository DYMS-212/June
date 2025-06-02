# src/simulators/vqe_simulator.py
import os
import sys
import time
import io
import logging
from contextlib import redirect_stdout
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, SPSA, COBYLA

from .base_simulator import BaseSimulator, SimulationConfig, SimulationResult
from ..utils.logger import get_logger


@dataclass
class VQESimulationConfig(SimulationConfig):
    """VQE仿真专用配置"""
    optimizer: str = "L_BFGS_B"
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    verbose: bool = False
    suppress_optimizer_output: bool = True
    optimizer_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 设置默认的优化器选项
        if not self.optimizer_options:
            if self.optimizer == "L_BFGS_B":
                self.optimizer_options = {
                    'disp': False,
                    'iprint': -1,
                    'maxiter': self.max_iterations
                }
            elif self.optimizer == "SPSA":
                self.optimizer_options = {
                    'maxiter': self.max_iterations
                }
            elif self.optimizer == "COBYLA":
                self.optimizer_options = {
                    'maxiter': self.max_iterations,
                    'disp': False
                }


class SilentOptimizationFilter(logging.Filter):
    """静默优化器输出的日志过滤器"""
    
    def filter(self, record):
        msg = record.getMessage()
        # 过滤包含优化器输出的消息
        return not any(text in msg for text in [
            'Found optimal point',
            'parameters',
            'Parameter',
            'iter',
            'optimization',
            'Optimizer',
            'optimal'
        ])


class VQESimulator(BaseSimulator):
    """VQE仿真器"""
    
    def __init__(self, 
                 config: VQESimulationConfig,
                 storage_manager,
                 hamiltonian,
                 **kwargs):
        
        # 确保配置是正确的类型
        if not isinstance(config, VQESimulationConfig):
            config = VQESimulationConfig(**config.__dict__)
        
        super().__init__(config, storage_manager, hamiltonian, **kwargs)
        
        # VQE特定的设置
        self.estimator = None
        self.optimizer = None
        
        # 用于抑制输出的设置
        self.null_io = NullIO()
        self.original_loggers = {}
        
    def setup(self, **kwargs):
        """设置VQE仿真器"""
        # 创建估计器
        self.estimator = Estimator()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 设置日志抑制
        if self.config.suppress_optimizer_output:
            self._setup_logging_suppression()
        
        self.logger.info(f"VQE simulator setup complete: optimizer={self.config.optimizer}")
    
    def _create_optimizer(self):
        """创建优化器实例"""
        options = self.config.optimizer_options.copy()
        
        if self.config.optimizer == "L_BFGS_B":
            return L_BFGS_B(options=options)
        elif self.config.optimizer == "SPSA":
            return SPSA(**options)
        elif self.config.optimizer == "COBYLA":
            return COBYLA(**options)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_logging_suppression(self):
        """设置日志抑制"""
        # 需要抑制的日志记录器
        logger_names = [
            'qiskit',
            'qiskit_algorithms',
            'qiskit.algorithms',
            'qiskit.primitives',
            'qiskit.algorithms.minimum_eigensolvers.vqe',
            'qiskit.algorithms.optimizers'
        ]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            self.original_loggers[logger_name] = logger.level
            logger.setLevel(logging.CRITICAL)
            logger.addFilter(SilentOptimizationFilter())
    
    def _restore_logging(self):
        """恢复日志设置"""
        for logger_name, original_level in self.original_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)
            # 移除过滤器
            logger.filters = [f for f in logger.filters if not isinstance(f, SilentOptimizationFilter)]
    
    def simulate_single(self, 
                       circuit: Any, 
                       global_idx: int, 
                       batch_idx: int, 
                       batch_inner_idx: int,
                       **kwargs) -> SimulationResult:
        """仿真单个电路的VQE"""
        
        start_time = time.time()
        
        try:
            # 确保circuit是QuantumCircuit对象
            if not isinstance(circuit, QuantumCircuit):
                circuit = QuantumCircuit.from_qasm_str(circuit)
            
            # 抑制输出（如果配置了）
            original_stdout = sys.stdout
            if self.config.suppress_optimizer_output and not self.config.verbose:
                sys.stdout = self.null_io
            
            result_data = {}
            
            try:
                # 使用重定向来抑制所有可能的输出
                with redirect_stdout(io.StringIO()):
                    # 创建VQE实例
                    vqe = VQE(
                        estimator=self.estimator,
                        ansatz=circuit,
                        optimizer=self.optimizer,
                        callback=self._empty_callback
                    )
                    
                    # 运行VQE
                    vqe_result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
                
                # 提取结果数据
                energy = vqe_result.eigenvalue.real
                optimal_parameters = vqe_result.optimal_parameters
                optimal_point = vqe_result.optimal_point
                cost_function_evals = vqe_result.cost_function_evals
                
                result_data = {
                    'energy': energy,
                    'optimal_parameters': optimal_parameters,
                    'optimal_point': optimal_point.tolist() if optimal_point is not None else None,
                    'cost_function_evals': cost_function_evals,
                    'optimizer_time': vqe_result.optimizer_time if hasattr(vqe_result, 'optimizer_time') else None
                }
                
                # 如果启用了详细输出，记录关键信息
                if self.config.verbose:
                    self.logger.debug(
                        f"Circuit {global_idx}: Energy = {energy:.6f}, "
                        f"Evaluations = {cost_function_evals}"
                    )
                
                status = "success"
                error_message = None
                
            except Exception as e:
                import traceback
                status = "error"
                error_message = f"{str(e)}\n{traceback.format_exc()}"
                result_data = {
                    'energy': None,
                    'optimal_parameters': None,
                    'optimal_point': None,
                    'cost_function_evals': None
                }
                
            finally:
                # 恢复stdout
                sys.stdout = original_stdout
            
            time_taken = time.time() - start_time
            
            # 创建结果对象
            simulation_result = SimulationResult(
                global_index=global_idx,
                batch_index=batch_idx,
                batch_inner_index=batch_inner_idx,
                status=status,
                time_taken=time_taken,
                result_data=result_data,
                error_message=error_message,
                metadata={
                    'circuit_depth': circuit.depth(),
                    'circuit_width': circuit.num_qubits,
                    'num_parameters': len(circuit.parameters)
                }
            )
            
            return simulation_result
            
        except Exception as e:
            # 处理意外错误
            time_taken = time.time() - start_time
            return SimulationResult(
                global_index=global_idx,
                batch_index=batch_idx,
                batch_inner_index=batch_inner_idx,
                status="error",
                time_taken=time_taken,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _empty_callback(self, *args, **kwargs):
        """空回调函数，用于抑制优化器输出"""
        pass
    
    def _is_better_result(self, new_result: SimulationResult, current_best: SimulationResult) -> bool:
        """判断新结果是否比当前最好结果更好"""
        # 对于VQE，更低的能量更好
        if new_result.status == "success" and current_best.status == "success":
            new_energy = new_result.result_data.get('energy')
            current_energy = current_best.result_data.get('energy')
            
            if new_energy is not None and current_energy is not None:
                return new_energy < current_energy
        
        # 回退到基类的逻辑
        return super()._is_better_result(new_result, current_best)
    
    def get_energy_statistics(self) -> Dict[str, Any]:
        """获取能量统计信息"""
        all_energies = []
        
        for batch in self.batch_results:
            for result in batch:
                if result.status == "success" and result.result_data.get('energy') is not None:
                    all_energies.append(result.result_data['energy'])
        
        if not all_energies:
            return {"message": "No successful energy calculations"}
        
        import numpy as np
        
        return {
            "count": len(all_energies),
            "min_energy": float(np.min(all_energies)),
            "max_energy": float(np.max(all_energies)),
            "mean_energy": float(np.mean(all_energies)),
            "median_energy": float(np.median(all_energies)),
            "std_energy": float(np.std(all_energies))
        }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        all_evals = []
        all_times = []
        
        for batch in self.batch_results:
            for result in batch:
                if result.status == "success":
                    evals = result.result_data.get('cost_function_evals')
                    if evals is not None:
                        all_evals.append(evals)
                    
                    all_times.append(result.time_taken)
        
        if not all_evals:
            return {"message": "No successful optimizations"}
        
        import numpy as np
        
        stats = {
            "function_evaluations": {
                "count": len(all_evals),
                "min": int(np.min(all_evals)),
                "max": int(np.max(all_evals)),
                "mean": float(np.mean(all_evals)),
                "median": float(np.median(all_evals))
            },
            "computation_time": {
                "count": len(all_times),
                "min": float(np.min(all_times)),
                "max": float(np.max(all_times)),
                "mean": float(np.mean(all_times)),
                "total": float(np.sum(all_times))
            }
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        try:
            # 恢复日志设置
            if hasattr(self, '_restore_logging'):
                self._restore_logging()
            
            # 清理其他资源
            if hasattr(self, 'estimator'):
                self.estimator = None
            if hasattr(self, 'optimizer'):
                self.optimizer = None
            
            self.logger.info("VQE simulator cleanup complete")
            
        except Exception as e:
            # 即使清理失败也要继续，但记录错误
            if hasattr(self, 'logger'):
                self.logger.warning(f"VQE simulator cleanup had issues: {e}")
            else:
                print(f"VQE simulator cleanup had issues: {e}")
class NullIO:
    """空输出设备，用于抑制输出"""
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass


# 便捷函数，用于创建VQE仿真器
def create_vqe_simulator(hamiltonian,
                        storage_manager,
                        batch_size: int = 100,
                        max_workers: Optional[int] = None,
                        n_repeat: int = 1,
                        optimizer: str = "L_BFGS_B",
                        verbose: bool = False,
                        **kwargs) -> VQESimulator:
    """创建VQE仿真器的便捷函数"""
    
    config = VQESimulationConfig(
        name="vqe",
        batch_size=batch_size,
        max_workers=max_workers,
        n_repeat=n_repeat,
        optimizer=optimizer,
        verbose=verbose,
        **kwargs
    )
    
    return VQESimulator(config, storage_manager, hamiltonian)


# 用于向后兼容的包装函数
def run_vqe_simulation(circuits: List[QuantumCircuit],
                      hamiltonian,
                      storage_manager,
                      config: Dict[str, Any]) -> Dict[str, Any]:
    """运行VQE仿真的简化接口"""
    
    vqe_config = VQESimulationConfig(**config)
    simulator = VQESimulator(vqe_config, storage_manager, hamiltonian)
    
    # 可选：创建进度跟踪器
    from ..utils.progress import create_progress_tracker
    progress_tracker = create_progress_tracker(
        "vqe_simulation", 
        len(circuits), 
        "Running VQE simulation"
    )
    
    try:
        results = simulator.run_simulation(circuits, progress_tracker)
        progress_tracker.complete()
        return results
    except Exception as e:
        progress_tracker.fail(str(e))
        raise
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    # 测试VQE仿真器
    from ..models.tfim import TFIM
    from ..utils.logger import setup_global_logger
    from ..utils.storage import StorageManager
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    
    # 设置日志和存储
    setup_global_logger({'name': 'test_vqe', 'level': 'INFO'})
    storage = StorageManager(base_dir="test_vqe_data")
    
    # 创建测试哈密顿量
    tfim = TFIM(size=4, J=1.0, g=1.0)
    hamiltonian, exact_energy = tfim.get_hamiltonian_and_energy()
    
    # 创建测试电路
    def create_test_circuit(n_qubits=4):
        qc = QuantumCircuit(n_qubits)
        
        # 添加参数化层
        for i in range(n_qubits):
            qc.h(i)
            param = Parameter(f"theta_{i}")
            qc.rz(param, i)
        
        # 添加纠缠层
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
        
        return qc
    
    test_circuits = [create_test_circuit() for _ in range(5)]
    
    # 创建VQE仿真器
    config = VQESimulationConfig(
        name="test_vqe",
        batch_size=3,
        n_repeat=2,
        optimizer="L_BFGS_B",
        verbose=True
    )
    
    simulator = VQESimulator(config, storage, hamiltonian)
    
    # 运行仿真
    print(f"精确基态能量: {exact_energy:.6f}")
    results = simulator.run_simulation(test_circuits)
    
    # 显示结果
    print(f"仿真完成: {results['success_count']}/{results['total_circuits']} 成功")
    
    # 获取统计信息
    energy_stats = simulator.get_energy_statistics()
    print(f"能量统计: {energy_stats}")
    
    opt_stats = simulator.get_optimization_statistics()
    print(f"优化统计: {opt_stats}")
    
    # 清理
    simulator.cleanup()
    
    print("VQE仿真器测试完成")
