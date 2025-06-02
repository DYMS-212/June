# src/simulators/expressibility_simulator.py
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from qiskit import QuantumCircuit

from .base_simulator import BaseSimulator, SimulationConfig, SimulationResult
from ..utils.logger import get_logger


@dataclass
class ExpressibilityConfig(SimulationConfig):
    """表达性计算专用配置"""
    samples: int = 5000
    bins: int = 75
    parallel_computation: bool = True
    random_seed: Optional[int] = None


class ExpressibilitySimulator(BaseSimulator):
    """表达性计算仿真器"""
    
    def __init__(self, 
                 config: ExpressibilityConfig,
                 storage_manager,
                 **kwargs):
        
        # 确保配置是正确的类型
        if not isinstance(config, ExpressibilityConfig):
            config = ExpressibilityConfig(**config.__dict__)
        
        # 表达性计算不需要哈密顿量
        super().__init__(config, storage_manager, hamiltonian=None, **kwargs)
    
    def setup(self, **kwargs):
        """设置表达性仿真器"""
        # 设置随机种子
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        self.logger.info(f"Expressibility simulator setup complete: "
                        f"samples={self.config.samples}, bins={self.config.bins}")
    
    def simulate_single(self, 
                       circuit: Any, 
                       global_idx: int, 
                       batch_idx: int, 
                       batch_inner_idx: int,
                       **kwargs) -> SimulationResult:
        """计算单个电路的表达性"""
        
        start_time = time.time()
        
        try:
            # 确保circuit是QuantumCircuit对象
            if not isinstance(circuit, QuantumCircuit):
                circuit = QuantumCircuit.from_qasm_str(circuit)
            
            # 检查电路是否有参数
            if len(circuit.parameters) == 0:
                return SimulationResult(
                    global_index=global_idx,
                    batch_index=batch_idx,
                    batch_inner_index=batch_inner_idx,
                    status="error",
                    time_taken=time.time() - start_time,
                    error_message="Circuit has no parameters for expressibility calculation"
                )
            
            # 计算表达性
            expressibility_value = self._calculate_expressibility(circuit)
            
            time_taken = time.time() - start_time
            
            result_data = {
                'expressibility': expressibility_value,
                'samples_used': self.config.samples,
                'bins_used': self.config.bins
            }
            
            return SimulationResult(
                global_index=global_idx,
                batch_index=batch_idx,
                batch_inner_index=batch_inner_idx,
                status="success",
                time_taken=time_taken,
                result_data=result_data,
                metadata={
                    'circuit_depth': circuit.depth(),
                    'circuit_width': circuit.num_qubits,
                    'num_parameters': len(circuit.parameters)
                }
            )
            
        except Exception as e:
            import traceback
            time_taken = time.time() - start_time
            return SimulationResult(
                global_index=global_idx,
                batch_index=batch_idx,
                batch_inner_index=batch_inner_idx,
                status="error",
                time_taken=time_taken,
                error_message=f"Expressibility calculation failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    def _calculate_expressibility(self, circuit: QuantumCircuit) -> float:
        """
        计算量子电路的表达性
        
        表达性衡量参数化量子电路产生的状态分布与Haar随机分布的接近程度
        """
        from qiskit.quantum_info import state_fidelity, Statevector
        
        qubits = circuit.num_qubits
        n_params = len(circuit.parameters)
        
        # 定义bin边界
        unit = 1.0 / self.config.bins
        limits = [unit * i for i in range(1, self.config.bins + 1)]
        
        # 初始化频率计数
        frequencies = np.zeros(self.config.bins)
        
        # 采样随机电路对并计算保真度
        for _ in range(self.config.samples):
            # 生成两组随机参数
            params_1 = np.random.uniform(0, 2 * np.pi, n_params)
            params_2 = np.random.uniform(0, 2 * np.pi, n_params)
            
            # 创建参数化电路
            circuit_1 = circuit.assign_parameters(params_1)
            circuit_2 = circuit.assign_parameters(params_2)
            
            # 计算状态保真度
            state_1 = Statevector(circuit_1)
            state_2 = Statevector(circuit_2)
            fidelity = state_fidelity(state_1, state_2)
            
            # 将保真度分配到对应的bin
            for j in range(self.config.bins):
                if fidelity <= limits[j]:
                    frequencies[j] += 1
                    break
        
        # 归一化频率为概率分布
        probabilities = frequencies / self.config.samples
        
        # 计算理论Haar分布
        bin_centers = [limit - (unit / 2) for limit in limits]
        p_haar_values = [self._p_haar(qubits, center) / self.config.bins for center in bin_centers]
        
        # 计算KL散度
        kl_divergence = self._kl_divergence(probabilities, p_haar_values)
        
        return kl_divergence
    
    def _p_haar(self, n_qubits: int, fidelity: float) -> float:
        """
        计算Haar随机状态的保真度概率密度函数
        
        对于n量子比特系统：P(F) = (2^n - 1) * (1 - F)^(2^n - 2)
        """
        if fidelity == 1:
            return 0
        else:
            N = 2 ** n_qubits
            return (N - 1) * ((1 - fidelity) ** (N - 2))
    
    def _kl_divergence(self, P: np.ndarray, Q: List[float]) -> float:
        """
        计算两个概率分布之间的KL散度
        
        KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
        """
        epsilon = 1e-8
        kl_div = 0.0
        
        for p, q in zip(P, Q):
            if p > 0:  # 只考虑非零概率
                kl_div += p * np.log((p + epsilon) / (q + epsilon))
        
        return abs(kl_div)
    
    def _is_better_result(self, new_result: SimulationResult, current_best: SimulationResult) -> bool:
        """判断新结果是否比当前最好结果更好"""
        # 对于表达性计算，较低的KL散度更好（更接近Haar分布）
        if new_result.status == "success" and current_best.status == "success":
            new_expr = new_result.result_data.get('expressibility')
            current_expr = current_best.result_data.get('expressibility')
            
            if new_expr is not None and current_expr is not None:
                return new_expr < current_expr
        
        # 回退到基类的逻辑
        return super()._is_better_result(new_result, current_best)
    
    def get_expressibility_statistics(self) -> Dict[str, Any]:
        """获取表达性统计信息"""
        all_expressibilities = []
        
        for batch in self.batch_results:
            for result in batch:
                if result.status == "success" and result.result_data.get('expressibility') is not None:
                    all_expressibilities.append(result.result_data['expressibility'])
        
        if not all_expressibilities:
            return {"message": "No successful expressibility calculations"}
        
        return {
            "count": len(all_expressibilities),
            "min_expressibility": float(np.min(all_expressibilities)),
            "max_expressibility": float(np.max(all_expressibilities)),
            "mean_expressibility": float(np.mean(all_expressibilities)),
            "median_expressibility": float(np.median(all_expressibilities)),
            "std_expressibility": float(np.std(all_expressibilities)),
            "config": {
                "samples": self.config.samples,
                "bins": self.config.bins
            }
        }
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("Expressibility simulator cleanup complete")


# 便捷函数，用于创建表达性仿真器
def create_expressibility_simulator(storage_manager,
                                   batch_size: int = 100,
                                   max_workers: Optional[int] = None,
                                   samples: int = 5000,
                                   bins: int = 75,
                                   **kwargs) -> ExpressibilitySimulator:
    """创建表达性仿真器的便捷函数"""
    
    config = ExpressibilityConfig(
        name="expressibility",
        batch_size=batch_size,
        max_workers=max_workers,
        samples=samples,
        bins=bins,
        **kwargs
    )
    
    return ExpressibilitySimulator(config, storage_manager)


# 用于向后兼容的包装函数
def run_expressibility_calculation(circuits: List[QuantumCircuit],
                                  storage_manager,
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """运行表达性计算的简化接口"""
    
    expr_config = ExpressibilityConfig(**config)
    simulator = ExpressibilitySimulator(expr_config, storage_manager)
    
    # 可选：创建进度跟踪器
    from ..utils.progress import create_progress_tracker
    progress_tracker = create_progress_tracker(
        "expressibility_calculation", 
        len(circuits), 
        "Calculating expressibility"
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


# 辅助函数：可视化表达性分布
def visualize_expressibility_distribution(expressibility_values: List[float], 
                                         output_path: Optional[str] = None,
                                         title: str = "Expressibility Distribution"):
    """可视化表达性分布"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(expressibility_values, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Expressibility (KL Divergence)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(alpha=0.3)
        
        # 添加统计信息
        mean_expr = np.mean(expressibility_values)
        std_expr = np.std(expressibility_values)
        plt.axvline(mean_expr, color='red', linestyle='--', 
                   label=f'Mean: {mean_expr:.4f}')
        plt.axvline(mean_expr - std_expr, color='orange', linestyle=':', alpha=0.7)
        plt.axvline(mean_expr + std_expr, color='orange', linestyle=':', alpha=0.7,
                   label=f'±1σ: {std_expr:.4f}')
        
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Expressibility distribution plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib is required for visualization. Install with 'pip install matplotlib'")


if __name__ == "__main__":
    # 测试表达性仿真器
    from ..utils.logger import setup_global_logger
    from ..utils.storage import StorageManager
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    
    # 设置日志和存储
    setup_global_logger({'name': 'test_expressibility', 'level': 'INFO'})
    storage = StorageManager(base_dir="test_expressibility_data")
    
    # 创建测试电路
    def create_test_circuit(n_qubits=4, depth=3):
        qc = QuantumCircuit(n_qubits)
        param_count = 0
        
        for layer in range(depth):
            # 参数化旋转门
            for i in range(n_qubits):
                param = Parameter(f"theta_{param_count}")
                qc.ry(param, i)
                param_count += 1
            
            # 纠缠层
            for i in range(n_qubits-1):
                qc.cx(i, i+1)
        
        return qc
    
    # 创建不同复杂度的测试电路
    test_circuits = [
        create_test_circuit(4, 2),  # 简单电路
        create_test_circuit(4, 4),  # 中等复杂度
        create_test_circuit(4, 6),  # 较复杂
    ]
    
    # 创建表达性仿真器
    config = ExpressibilityConfig(
        name="test_expressibility",
        batch_size=2,
        samples=1000,  # 较少样本用于测试
        bins=50
    )
    
    simulator = ExpressibilitySimulator(config, storage)
    
    # 运行表达性计算
    print("开始表达性计算测试...")
    results = simulator.run_simulation(test_circuits)
    
    # 显示结果
    print(f"表达性计算完成: {results['success_count']}/{results['total_circuits']} 成功")
    
    # 获取统计信息
    expr_stats = simulator.get_expressibility_statistics()
    print(f"表达性统计: {expr_stats}")
    
    # 提取表达性值进行可视化
    expr_values = []
    for batch in simulator.batch_results:
        for result in batch:
            if result.status == "success":
                expr_values.append(result.result_data['expressibility'])
    
    if expr_values:
        print(f"表达性值: {expr_values}")
        # 如果有matplotlib，可以可视化
        try:
            visualize_expressibility_distribution(
                expr_values, 
                "test_expressibility_distribution.png",
                "Test Expressibility Distribution"
            )
        except:
            print("跳过可视化（matplotlib不可用）")
    
    # 清理
    simulator.cleanup()
    
    print("表达性仿真器测试完成")
