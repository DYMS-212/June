# src/simulators/noisy_vqe_simulator.py
"""
含噪声VQE仿真器 - 基于原始 noise_vqe.py 适配到新框架
"""
import os
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from .base_simulator import BaseSimulator, SimulationConfig, SimulationResult
from ..utils.logger import get_logger


@dataclass
class NoisyVQEConfig(SimulationConfig):
    """含噪声VQE仿真专用配置"""
    # 噪声模型参数
    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    t1_time: float = 50e-6  # 50微秒
    t2_time: float = 20e-6  # 20微秒
    gate_time_1q: float = 50e-9  # 50纳秒
    gate_time_2q: float = 200e-9  # 200纳秒
    
    # 量子后端设置
    shot_count: int = 1024
    backend_method: str = "density_matrix"
    
    # 优化器设置
    optimizer: str = "L_BFGS_B"
    random_seed: int = 42
    
    # 耦合图设置 (None表示全连接)
    coupling_map: Optional[List[tuple]] = None


class NoisyVQESimulator(BaseSimulator):
    """含噪声VQE仿真器"""
    
    def __init__(self, 
                 config: NoisyVQEConfig,
                 storage_manager,
                 hamiltonian,
                 reference_results: Optional[Dict] = None,
                 **kwargs):
        
        # 确保配置是正确的类型
        if not isinstance(config, NoisyVQEConfig):
            config = NoisyVQEConfig(**config.__dict__)
        
        super().__init__(config, storage_manager, hamiltonian, **kwargs)
        
        # 保存参考结果（无噪声VQE结果）用于比较
        self.reference_results = reference_results or {}
        
        # 噪声VQE特定设置
        self.noise_model = None
        self.coupling_map = None
        self.noisy_estimator = None
        self.optimizer = None
    
    def setup(self, **kwargs):
        """设置含噪声VQE仿真器"""
        # 创建噪声模型
        self.noise_model = self._create_noise_model()
        
        # 设置耦合图
        self.coupling_map = self._create_coupling_map()
        
        # 创建带噪声的估计器
        self.noisy_estimator = self._create_noisy_estimator()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        self.logger.info(f"Noisy VQE simulator setup complete: "
                        f"noise_model={type(self.noise_model).__name__}, "
                        f"optimizer={self.config.optimizer}")
    
    def _create_noise_model(self) -> NoiseModel:
        """创建噪声模型 - 基于原始 noise_vqe.py 的实现"""
        noise_model = NoiseModel()
        
        # 单量子比特门的噪声
        single_qubit_gates = ['rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't', 'u1', 'u2', 'u3']
        
        for gate in single_qubit_gates:
            # 去极化错误
            depol_error = depolarizing_error(self.config.single_qubit_error, 1)
            
            # 热弛豫错误
            thermal_error = thermal_relaxation_error(
                self.config.t1_time, 
                self.config.t2_time, 
                self.config.gate_time_1q
            )
            
            # 组合错误
            combined_error = thermal_error.compose(depol_error)
            
            # 添加到所有量子比特
            noise_model.add_all_qubit_quantum_error(combined_error, gate)
        
        # 双量子比特门的噪声
        two_qubit_gates = ['cx', 'cy', 'cz', 'cnot', 'cphase', 'rxx', 'ryy', 'rzz']
        
        for gate in two_qubit_gates:
            # 去极化错误
            depol_error = depolarizing_error(self.config.two_qubit_error, 2)
            
            # 热弛豫错误（应用到两个量子比特）
            thermal_error_q0 = thermal_relaxation_error(
                self.config.t1_time, 
                self.config.t2_time, 
                self.config.gate_time_2q
            )
            thermal_error_q1 = thermal_relaxation_error(
                self.config.t1_time, 
                self.config.t2_time, 
                self.config.gate_time_2q
            )
            thermal_error_2q = thermal_error_q0.tensor(thermal_error_q1)
            
            # 组合错误
            combined_error = thermal_error_2q.compose(depol_error)
            
            # 添加到所有量子比特对
            noise_model.add_all_qubit_quantum_error(combined_error, gate)
        
        return noise_model
    
    def _create_coupling_map(self) -> List[tuple]:
        """创建耦合图"""
        if self.config.coupling_map is not None:
            return self.config.coupling_map
        
        # 从哈密顿量推断量子比特数量
        # 这里假设我们有一个方法来获取量子比特数量
        num_qubits = self._get_num_qubits_from_hamiltonian()
        
        # 创建全连接耦合图
        coupling_map = []
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                coupling_map.append((i, j))
        
        return coupling_map
    
    def _get_num_qubits_from_hamiltonian(self) -> int:
        """从哈密顿量获取量子比特数量"""
        # 这里我们需要从哈密顿量推断量子比特数量
        # 对于TFIM模型，可以从哈密顿量的维度推断
        if hasattr(self.hamiltonian, 'num_qubits'):
            return self.hamiltonian.num_qubits
        else:
            # 从哈密顿量矩阵维度推断
            import numpy as np
            dim = self.hamiltonian.to_matrix().shape[0]
            return int(np.log2(dim))
    
    def _create_noisy_estimator(self) -> AerEstimator:
        """创建带噪声的估计器"""
        return AerEstimator(options={
            "default_precision": 1e-2,
            "backend_options": {
                "method": self.config.backend_method,
                "coupling_map": self.coupling_map,
                "noise_model": self.noise_model,
            },
            "run_options": {"seed": self.config.random_seed},
        })
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer == "L_BFGS_B":
            return L_BFGS_B()
        elif self.config.optimizer == "SPSA":
            return SPSA()
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def simulate_single(self, 
                       circuit: Any, 
                       global_idx: int, 
                       batch_idx: int, 
                       batch_inner_idx: int,
                       **kwargs) -> SimulationResult:
        """仿真单个电路的含噪声VQE"""
        
        start_time = time.time()
        
        try:
            # 确保circuit是QuantumCircuit对象
            if not isinstance(circuit, QuantumCircuit):
                circuit = QuantumCircuit.from_qasm_str(circuit)
            
            # 设置随机种子
            algorithm_globals.random_seed = self.config.random_seed
            
            # 运行含噪声VQE
            vqe = VQE(self.noisy_estimator, circuit, optimizer=self.optimizer)
            result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)
            
            noisy_energy = result.eigenvalue.real
            
            # 获取参考能量（如果有的话）
            reference_energy = None
            noise_error = None
            
            if self.reference_results:
                # 尝试从参考结果中找到对应的无噪声能量
                ref_key = str(global_idx)
                if ref_key in self.reference_results:
                    reference_energy = self.reference_results[ref_key]
                    noise_error = abs(noisy_energy - reference_energy)
            
            time_taken = time.time() - start_time
            
            result_data = {
                'noisy_energy': noisy_energy,
                'reference_energy': reference_energy,
                'noise_error': noise_error,
                'optimal_parameters': result.optimal_parameters,
                'cost_function_evals': result.cost_function_evals,
                'noise_config': {
                    'single_qubit_error': self.config.single_qubit_error,
                    'two_qubit_error': self.config.two_qubit_error,
                    't1_time': self.config.t1_time,
                    't2_time': self.config.t2_time
                }
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
                error_message=f"Noisy VQE failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    def _is_better_result(self, new_result: SimulationResult, current_best: SimulationResult) -> bool:
        """判断新结果是否比当前最好结果更好"""
        # 对于含噪声VQE，更低的能量更好
        if new_result.status == "success" and current_best.status == "success":
            new_energy = new_result.result_data.get('noisy_energy')
            current_energy = current_best.result_data.get('noisy_energy')
            
            if new_energy is not None and current_energy is not None:
                return new_energy < current_energy
        
        # 回退到基类的逻辑
        return super()._is_better_result(new_result, current_best)
    
    def get_noise_statistics(self) -> Dict[str, Any]:
        """获取噪声影响统计信息"""
        all_noisy_energies = []
        all_reference_energies = []
        all_noise_errors = []
        
        for batch in self.batch_results:
            for result in batch:
                if result.status == "success":
                    data = result.result_data
                    if data.get('noisy_energy') is not None:
                        all_noisy_energies.append(data['noisy_energy'])
                    
                    if data.get('reference_energy') is not None:
                        all_reference_energies.append(data['reference_energy'])
                    
                    if data.get('noise_error') is not None:
                        all_noise_errors.append(data['noise_error'])
        
        if not all_noisy_energies:
            return {"message": "No successful noisy VQE calculations"}
        
        stats = {
            "noisy_energies": {
                "count": len(all_noisy_energies),
                "min": float(np.min(all_noisy_energies)),
                "max": float(np.max(all_noisy_energies)),
                "mean": float(np.mean(all_noisy_energies)),
                "std": float(np.std(all_noisy_energies))
            }
        }
        
        if all_reference_energies:
            stats["reference_energies"] = {
                "count": len(all_reference_energies),
                "min": float(np.min(all_reference_energies)),
                "max": float(np.max(all_reference_energies)),
                "mean": float(np.mean(all_reference_energies)),
                "std": float(np.std(all_reference_energies))
            }
        
        if all_noise_errors:
            stats["noise_impact"] = {
                "mean_error": float(np.mean(all_noise_errors)),
                "max_error": float(np.max(all_noise_errors)),
                "min_error": float(np.min(all_noise_errors)),
                "std_error": float(np.std(all_noise_errors))
            }
        
        stats["noise_config"] = {
            "single_qubit_error": self.config.single_qubit_error,
            "two_qubit_error": self.config.two_qubit_error,
            "t1_time": self.config.t1_time,
            "t2_time": self.config.t2_time
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.noisy_estimator = None
        self.noise_model = None
        self.optimizer = None
        
        self.logger.info("Noisy VQE simulator cleanup complete")


# 便捷函数，用于创建含噪声VQE仿真器
def create_noisy_vqe_simulator(hamiltonian,
                              storage_manager,
                              reference_results: Optional[Dict] = None,
                              batch_size: int = 50,
                              max_workers: Optional[int] = None,
                              single_qubit_error: float = 0.001,
                              two_qubit_error: float = 0.01,
                              **kwargs) -> NoisyVQESimulator:
    """创建含噪声VQE仿真器的便捷函数"""
    
    config = NoisyVQEConfig(
        name="noisy_vqe",
        batch_size=batch_size,
        max_workers=max_workers,
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        **kwargs
    )
    
    return NoisyVQESimulator(config, storage_manager, hamiltonian, reference_results)


if __name__ == "__main__":
    # 测试含噪声VQE仿真器
    print("含噪声VQE仿真器加载完成")
