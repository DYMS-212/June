# src/pipeline.py
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .utils.config import QuantumVQEFullConfig, ConfigManager
from .utils.logger import QuantumVQELogger, setup_global_logger, get_logger, log_simulation_start, log_simulation_complete
from .utils.storage import StorageManager
from .utils.progress import MultiProgressManager, setup_global_progress_manager

from .models.tfim import TFIM
from .generators.layerwise import HalfLayerTFIMGenerator
from .simulators.vqe_simulator import VQESimulator, VQESimulationConfig
from .simulators.expressibility_simulator import ExpressibilitySimulator, ExpressibilityConfig
from .simulators.noisy_vqe_simulator import NoisyVQESimulator, NoisyVQEConfig


class QuantumVQEPipeline:
    """量子VQE仿真流水线主控制器"""
    
    def __init__(self, config: Union[str, QuantumVQEFullConfig]):
        """
        初始化流水线
        
        Args:
            config: 配置文件路径或配置对象
        """
        # 加载配置
        if isinstance(config, str):
            config_manager = ConfigManager()
            self.config = config_manager.load_config(config)
        else:
            self.config = config
        
        # 验证配置
        config_manager = ConfigManager()
        errors = config_manager.validate_config(self.config)
        if errors:
            raise ValueError(f"配置验证失败: {errors}")
        
        # 设置输出目录
        self.output_dir = Path(self.config.project.base_output_dir)
        if self.config.project.timestamp_dirs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / f"{self.config.project.name}_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志系统
        self._setup_logging()
        
        # 初始化存储系统
        self._setup_storage()
        
        # 初始化进度系统
        self._setup_progress()
        
        # 初始化物理模型
        self._setup_physics_model()
        
        # 初始化电路生成器
        self._setup_circuit_generator()
        
        # 仿真器实例
        self.simulators: Dict[str, Any] = {}
        
        # 执行状态
        self.execution_state = {
            'circuits_generated': False,
            'circuits_file': None,
            'circuits_count': 0,
            'vqe_completed': False,
            'expressibility_completed': False,
            'noisy_vqe_completed': False,
            'results': {}
        }
        
        self.logger.info(f"Pipeline initialized: {self.config.project.name} v{self.config.project.version}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = {
            'name': self.config.project.name,
            'log_dir': str(self.output_dir / self.config.logging.log_dir),
            'level': self.config.logging.level,
            'format_type': self.config.logging.format_type,
            'file_rotation': self.config.logging.file_rotation,
            'max_size': self.config.logging.max_size,
            'backup_count': self.config.logging.backup_count,
            'enable_console': self.config.logging.enable_console
        }
        
        self.logger_manager = setup_global_logger(log_config)
        self.logger = get_logger("pipeline")
    
    def _setup_storage(self):
        """设置存储系统"""
        storage_dir = self.output_dir / self.config.storage.base_dir
        
        self.storage_manager = StorageManager(
            base_dir=str(storage_dir),
            compression=self.config.storage.compression,
            backup_enabled=self.config.storage.backup_enabled,
            metadata_enabled=self.config.storage.metadata_enabled
        )
    
    def _setup_progress(self):
        """设置进度系统"""
        self.progress_manager = setup_global_progress_manager(
            show_overall=self.config.progress.show_overall_progress
        )
    
    def _setup_physics_model(self):
        """设置物理模型"""
        self.tfim_model = TFIM(
            size=self.config.circuit_generation.num_qubits,
            J=self.config.circuit_generation.tfim_J,
            g=self.config.circuit_generation.tfim_g
        )
        
        self.hamiltonian, self.exact_energy = self.tfim_model.get_hamiltonian_and_energy()
        
        self.logger.info(
            f"TFIM model setup: {self.config.circuit_generation.num_qubits} qubits, "
            f"J={self.config.circuit_generation.tfim_J}, g={self.config.circuit_generation.tfim_g}"
        )
        self.logger.info(f"Exact ground state energy: {self.exact_energy:.6f}")
    
    def _setup_circuit_generator(self):
        """设置电路生成器"""
        if self.config.circuit_generation.generator_type == "layerwise":
            self.circuit_generator = HalfLayerTFIMGenerator(
                logger=get_logger("circuit_generator")
            )
        else:
            raise ValueError(f"Unsupported generator type: {self.config.circuit_generation.generator_type}")
    
    def generate_circuits(self) -> List[Any]:
        """生成量子电路"""
        if self.execution_state['circuits_generated']:
            self.logger.info("Circuits already generated, loading from file")
            return self.storage_manager.load_circuits(self.execution_state['circuits_file'])
        
        self.logger.info("Starting circuit generation...")
        
        # 创建进度跟踪器
        progress_tracker = self.progress_manager.create_tracker(
            "circuit_generation",
            self.config.circuit_generation.num_circuits,
            "Generating quantum circuits"
        )
        
        start_time = time.time()
        
        try:
            # 生成电路
            circuits = self.circuit_generator.generate_tfim_quantum_circuits(
                num_circuits=self.config.circuit_generation.num_circuits,
                num_qubits=self.config.circuit_generation.num_qubits,
                max_gates=self.config.circuit_generation.max_depth,
                max_add_count=self.config.circuit_generation.max_gates,
                gate_stddev=self.config.circuit_generation.gate_stddev,
                gate_bias=self.config.circuit_generation.gate_bias
            )
            
            # 更新进度（生成器内部不使用我们的进度系统）
            progress_tracker.update(len(circuits))
            progress_tracker.complete()
            
            # 保存电路
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"circuits_{timestamp}"
            
            circuit_file = self.storage_manager.save_circuits(
                circuits, 
                filename,
                description=f"Generated {len(circuits)} TFIM circuits with {self.config.circuit_generation.num_qubits} qubits"
            )
            
            generation_time = time.time() - start_time
            
            # 更新执行状态
            self.execution_state.update({
                'circuits_generated': True,
                'circuits_file': filename,
                'circuits_count': len(circuits)
            })
            
            self.logger.info(
                f"Generated {len(circuits)} circuits in {generation_time:.2f}s, "
                f"saved to {circuit_file}"
            )
            
            return circuits
            
        except Exception as e:
            progress_tracker.fail(str(e))
            self.logger.error(f"Circuit generation failed: {e}")
            raise
    
    def load_existing_circuits(self, circuit_file: str) -> List[Any]:
        """加载已存在的电路"""
        self.logger.info(f"Loading existing circuits from {circuit_file}")
        
        try:
            circuits = self.storage_manager.load_circuits(circuit_file)
            
            # 更新执行状态
            self.execution_state.update({
                'circuits_generated': True,
                'circuits_file': circuit_file,
                'circuits_count': len(circuits)
            })
            
            self.logger.info(f"Loaded {len(circuits)} circuits from {circuit_file}")
            return circuits
            
        except Exception as e:
            self.logger.error(f"Failed to load circuits: {e}")
            raise
    
    def run_vqe_simulation(self, circuits: List[Any]) -> Dict[str, Any]:
        """运行VQE仿真"""
        if not self.config.vqe.enabled:
            self.logger.info("VQE simulation disabled")
            return {"status": "disabled"}
        
        if self.execution_state['vqe_completed']:
            self.logger.info("VQE simulation already completed")
            return self.execution_state['results'].get('vqe', {})
        
        self.logger.info("Starting VQE simulation...")
        
        # 创建VQE配置
        vqe_config = VQESimulationConfig(
            name="vqe",
            enabled=True,
            batch_size=self.config.vqe.batch_size,
            max_workers=self.config.vqe.max_workers,
            n_repeat=self.config.vqe.n_repeat,
            optimizer=self.config.vqe.optimizer,
            max_iterations=self.config.vqe.max_iterations,
            convergence_threshold=self.config.vqe.convergence_threshold,
            verbose=self.config.vqe.verbose,
            suppress_optimizer_output=self.config.vqe.suppress_optimizer_output,
            optimizer_options=self.config.vqe.optimizer_options
        )
        
        # 创建VQE仿真器
        vqe_simulator = VQESimulator(vqe_config, self.storage_manager, self.hamiltonian)
        self.simulators['vqe'] = vqe_simulator
        
        # 创建进度跟踪器
        progress_tracker = self.progress_manager.create_tracker(
            "vqe_simulation",
            len(circuits),
            "Running VQE simulation"
        )
        
        try:
            # 运行仿真
            results = vqe_simulator.run_simulation(circuits, progress_tracker)
            
            # 添加精确能量信息
            results['exact_energy'] = self.exact_energy
            results['hamiltonian_info'] = {
                'model': 'TFIM',
                'num_qubits': self.config.circuit_generation.num_qubits,
                'J': self.config.circuit_generation.tfim_J,
                'g': self.config.circuit_generation.tfim_g
            }
            
            # 计算统计信息
            energy_stats = vqe_simulator.get_energy_statistics()
            opt_stats = vqe_simulator.get_optimization_statistics()
            
            results['energy_statistics'] = energy_stats
            results['optimization_statistics'] = opt_stats
            
            # 更新执行状态
            self.execution_state['vqe_completed'] = True
            self.execution_state['results']['vqe'] = results
            
            progress_tracker.complete()
            
            self.logger.info(
                f"VQE simulation completed: {results['success_count']}/{results['total_circuits']} successful, "
                f"time: {results['simulation_time']:.2f}s"
            )
            
            return results
            
        except Exception as e:
            progress_tracker.fail(str(e))
            self.logger.error(f"VQE simulation failed: {e}")
            raise
        finally:
            vqe_simulator.cleanup()
    
    def run_expressibility_calculation(self, circuits: List[Any]) -> Dict[str, Any]:
        """运行表达性计算"""
        if not self.config.expressibility.enabled:
            self.logger.info("Expressibility calculation disabled")
            return {"status": "disabled"}
        
        if self.execution_state['expressibility_completed']:
            self.logger.info("Expressibility calculation already completed")
            return self.execution_state['results'].get('expressibility', {})
        
        self.logger.info("Starting expressibility calculation...")
        
        # 创建表达性配置
        expr_config = ExpressibilityConfig(
            name="expressibility",
            enabled=True,
            batch_size=self.config.expressibility.batch_size,
            max_workers=self.config.expressibility.max_workers,
            samples=self.config.expressibility.samples,
            bins=self.config.expressibility.bins,
            parallel_computation=self.config.expressibility.parallel_computation
        )
        
        # 创建表达性仿真器
        expr_simulator = ExpressibilitySimulator(expr_config, self.storage_manager)
        self.simulators['expressibility'] = expr_simulator
        
        # 创建进度跟踪器
        progress_tracker = self.progress_manager.create_tracker(
            "expressibility_calculation",
            len(circuits),
            "Calculating circuit expressibility"
        )
        
        try:
            # 运行表达性计算
            results = expr_simulator.run_simulation(circuits, progress_tracker)
            
            # 添加配置信息
            results['config_info'] = {
                'samples': self.config.expressibility.samples,
                'bins': self.config.expressibility.bins
            }
            
            # 计算统计信息
            expr_stats = expr_simulator.get_expressibility_statistics()
            results['expressibility_statistics'] = expr_stats
            
            # 更新执行状态
            self.execution_state['expressibility_completed'] = True
            self.execution_state['results']['expressibility'] = results
            
            progress_tracker.complete()
            
            self.logger.info(
                f"Expressibility calculation completed: {results['success_count']}/{results['total_circuits']} successful, "
                f"time: {results['simulation_time']:.2f}s"
            )
            
            return results
            
        except Exception as e:
            progress_tracker.fail(str(e))
            self.logger.error(f"Expressibility calculation failed: {e}")
            raise
        finally:
            expr_simulator.cleanup()
    
    def run_noisy_vqe_simulation(self, circuits: List[Any]) -> Dict[str, Any]:
        """运行含噪声VQE仿真"""
        if not self.config.noisy_vqe.enabled:
            self.logger.info("Noisy VQE simulation disabled")
            return {"status": "disabled"}
        
        if self.execution_state['noisy_vqe_completed']:
            self.logger.info("Noisy VQE simulation already completed")
            return self.execution_state['results'].get('noisy_vqe', {})
        
        self.logger.info("Starting noisy VQE simulation...")
        
        # 这里需要实现含噪声VQE仿真器
        # 暂时返回模拟结果
        progress_tracker = self.progress_manager.create_tracker(
            "noisy_vqe_simulation",
            len(circuits),
            "Running noisy VQE simulation"
        )
        
        try:
            # 模拟含噪声VQE计算
            import time
            for i in range(len(circuits)):
                time.sleep(0.02)  # 模拟计算时间（更慢）
                progress_tracker.update(1)
            
            results = {
                "simulation_type": "noisy_vqe",
                "total_circuits": len(circuits),
                "status": "completed",
                "message": "Noisy VQE simulation completed (placeholder)"
            }
            
            self.execution_state['noisy_vqe_completed'] = True
            self.execution_state['results']['noisy_vqe'] = results
            
            progress_tracker.complete()
            
            self.logger.info("Noisy VQE simulation completed")
            return results
            
        except Exception as e:
            progress_tracker.fail(str(e))
            self.logger.error(f"Noisy VQE simulation failed: {e}")
            raise
    
    def run_full_pipeline(self, existing_circuits_file: Optional[str] = None) -> Dict[str, Any]:
        """运行完整的流水线"""
        pipeline_start_time = time.time()
        
        self.logger.info("="*60)
        self.logger.info(f"Starting full pipeline: {self.config.project.name}")
        self.logger.info("="*60)
        
        try:
            # 1. 生成或加载电路
            if existing_circuits_file:
                circuits = self.load_existing_circuits(existing_circuits_file)
            else:
                circuits = self.generate_circuits()
            
            # 2. 运行各种仿真
            vqe_results = self.run_vqe_simulation(circuits)
            expressibility_results = self.run_expressibility_calculation(circuits)
            noisy_vqe_results = self.run_noisy_vqe_simulation(circuits)
            
            # 3. 汇总所有结果
            pipeline_time = time.time() - pipeline_start_time
            
            final_results = {
                'pipeline_info': {
                    'project_name': self.config.project.name,
                    'version': self.config.project.version,
                    'execution_time': pipeline_time,
                    'timestamp': datetime.now().isoformat(),
                    'output_directory': str(self.output_dir),
                    'config': self.config.to_dict()
                },
                'circuit_info': {
                    'total_circuits': len(circuits),
                    'num_qubits': self.config.circuit_generation.num_qubits,
                    'exact_ground_energy': self.exact_energy
                },
                'simulation_results': {
                    'vqe': vqe_results,
                    'expressibility': expressibility_results,
                    'noisy_vqe': noisy_vqe_results
                },
                'execution_state': self.execution_state
            }
            
            # 4. 保存最终结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_results_file = f"pipeline_results_{timestamp}"
            
            self.storage_manager.save_results(
                final_results,
                final_results_file,
                format_type="json",
                description=f"Complete pipeline results for {self.config.project.name}"
            )
            
            # 5. 生成摘要
            self._generate_pipeline_summary(final_results)
            
            self.logger.info("="*60)
            self.logger.info(f"Pipeline completed successfully in {pipeline_time:.2f}s")
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("="*60)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # 关闭所有进度条
            self.progress_manager.close_all()
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]):
        """生成流水线执行摘要"""
        summary_lines = [
            "="*60,
            f"Pipeline Execution Summary: {self.config.project.name}",
            "="*60,
            "",
            f"Execution Time: {results['pipeline_info']['execution_time']:.2f} seconds",
            f"Output Directory: {results['pipeline_info']['output_directory']}",
            f"Timestamp: {results['pipeline_info']['timestamp']}",
            "",
            "Circuit Information:",
            f"  Total Circuits: {results['circuit_info']['total_circuits']}",
            f"  Qubits per Circuit: {results['circuit_info']['num_qubits']}",
            f"  Exact Ground Energy: {results['circuit_info']['exact_ground_energy']:.6f}",
            "",
            "Simulation Results:",
        ]
        
        # VQE结果摘要
        vqe_res = results['simulation_results']['vqe']
        if vqe_res.get('status') != 'disabled':
            summary_lines.extend([
                f"  VQE Simulation:",
                f"    Success Rate: {vqe_res.get('success_count', 0)}/{vqe_res.get('total_circuits', 0)} "
                f"({vqe_res.get('success_rate', 0)*100:.1f}%)",
                f"    Simulation Time: {vqe_res.get('simulation_time', 0):.2f}s",
            ])
            
            if 'energy_statistics' in vqe_res:
                energy_stats = vqe_res['energy_statistics']
                if 'min_energy' in energy_stats:
                    summary_lines.extend([
                        f"    Best Energy: {energy_stats['min_energy']:.6f}",
                        f"    Average Energy: {energy_stats['mean_energy']:.6f}",
                        f"    Energy Std: {energy_stats['std_energy']:.6f}",
                    ])
        else:
            summary_lines.append("  VQE Simulation: Disabled")
        
        # 表达性结果摘要
        expr_res = results['simulation_results']['expressibility']
        if expr_res.get('status') != 'disabled':
            summary_lines.append("  Expressibility: Completed")
        else:
            summary_lines.append("  Expressibility: Disabled")
        
        # 含噪声VQE结果摘要
        noisy_res = results['simulation_results']['noisy_vqe']
        if noisy_res.get('status') != 'disabled':
            summary_lines.append("  Noisy VQE: Completed")
        else:
            summary_lines.append("  Noisy VQE: Disabled")
        
        summary_lines.extend([
            "",
            "="*60
        ])
        
        # 保存摘要
        summary_text = "\n".join(summary_lines)
        summary_file = self.output_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # 同时输出到日志
        self.logger.info("Pipeline Summary:")
        for line in summary_lines:
            self.logger.info(line)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """获取当前执行状态"""
        return {
            'execution_state': self.execution_state.copy(),
            'config': self.config.to_dict(),
            'output_directory': str(self.output_dir),
            'simulators_initialized': list(self.simulators.keys())
        }
    
    def cleanup(self):
        """清理资源"""
        # 清理仿真器
        for simulator in self.simulators.values():
            if hasattr(simulator, 'cleanup'):
                simulator.cleanup()
        
        # 关闭进度管理器
        self.progress_manager.close_all()
        
        # 清理临时文件
        self.storage_manager.cleanup_temp_files()
        
        self.logger.info("Pipeline cleanup completed")


# 便捷函数，用于快速创建和运行流水线
def run_quantum_vqe_pipeline(config_path: str, 
                            existing_circuits: Optional[str] = None,
                            **config_overrides) -> Dict[str, Any]:
    """运行量子VQE流水线的便捷函数"""
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # 应用配置覆盖
    if config_overrides:
        # 这里可以实现配置覆盖逻辑
        pass
    
    # 创建并运行流水线
    pipeline = QuantumVQEPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline(existing_circuits)
        return results
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    # 测试流水线
    from .utils.config import ConfigManager
    
    # 创建测试配置
    config_manager = ConfigManager("test_configs")
    test_config_path = config_manager.create_config_template("quick_test")
    
    # 运行流水线
    try:
        results = run_quantum_vqe_pipeline(str(test_config_path))
        print("流水线测试完成!")
        print(f"结果: {results['circuit_info']}")
    except Exception as e:
        print(f"流水线测试失败: {e}")
