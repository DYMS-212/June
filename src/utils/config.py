# src/utils/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .logger import get_logger


@dataclass
class ProjectConfig:
    """项目总配置"""
    name: str = "quantum_vqe_simulation"
    version: str = "1.0.0"
    description: str = "Quantum VQE Circuit Simulation Project"
    base_output_dir: str = "output"
    timestamp_dirs: bool = True


@dataclass
class CircuitGenerationConfig:
    """电路生成配置"""
    num_circuits: int = 5000
    num_qubits: int = 8
    max_depth: int = 50
    max_gates: int = 48
    gate_stddev: float = 1.35
    gate_bias: float = 0.5
    
    # TFIM模型参数
    tfim_J: float = 1.0
    tfim_g: float = 1.0
    
    # 生成器参数
    generator_type: str = "layerwise"  # layerwise, gatewise
    deduplication: bool = True


@dataclass
class VQEConfig:
    """VQE仿真配置"""
    enabled: bool = True
    batch_size: int = 100
    max_workers: Optional[int] = None
    n_repeat: int = 1
    
    # 优化器设置
    optimizer: str = "L_BFGS_B"
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    # 输出控制
    verbose: bool = False
    suppress_optimizer_output: bool = True
    
    # 优化器特定选项
    optimizer_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpressibilityConfig:
    """表达性计算配置"""
    enabled: bool = True
    samples: int = 5000
    bins: int = 75
    batch_size: int = 100
    max_workers: Optional[int] = None
    
    # 是否并行计算表达性
    parallel_computation: bool = True


@dataclass
class NoisyVQEConfig:
    """含噪声VQE配置"""
    enabled: bool = False
    batch_size: int = 50
    max_workers: Optional[int] = None
    
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


@dataclass
class StorageConfig:
    """存储配置"""
    base_dir: str = "data"
    compression: bool = True
    backup_enabled: bool = True
    metadata_enabled: bool = True
    
    # 文件格式设置
    circuit_format: str = "pkl"  # pkl, pkl.gz
    results_format: str = "json"  # json, pkl, hdf5
    metadata_format: str = "json"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format_type: str = "human"  # human, structured
    file_rotation: bool = True
    max_size: str = "100MB"
    backup_count: int = 5
    enable_console: bool = True
    
    # 日志目录
    log_dir: str = "logs"


@dataclass
class ProgressConfig:
    """进度显示配置"""
    show_overall_progress: bool = True
    show_individual_progress: bool = True
    log_interval: int = 10  # 每N%记录一次日志
    update_frequency: float = 1.0  # 更新频率（秒）


@dataclass
class QuantumVQEFullConfig:
    """完整的项目配置"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    circuit_generation: CircuitGenerationConfig = field(default_factory=CircuitGenerationConfig)
    vqe: VQEConfig = field(default_factory=VQEConfig)
    expressibility: ExpressibilityConfig = field(default_factory=ExpressibilityConfig)
    noisy_vqe: NoisyVQEConfig = field(default_factory=NoisyVQEConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumVQEFullConfig':
        """从字典创建配置"""
        # 为每个配置部分创建对应的dataclass实例
        config_kwargs = {}
        
        if 'project' in data:
            config_kwargs['project'] = ProjectConfig(**data['project'])
        
        if 'circuit_generation' in data:
            config_kwargs['circuit_generation'] = CircuitGenerationConfig(**data['circuit_generation'])
        
        if 'vqe' in data:
            config_kwargs['vqe'] = VQEConfig(**data['vqe'])
        
        if 'expressibility' in data:
            config_kwargs['expressibility'] = ExpressibilityConfig(**data['expressibility'])
        
        if 'noisy_vqe' in data:
            config_kwargs['noisy_vqe'] = NoisyVQEConfig(**data['noisy_vqe'])
        
        if 'storage' in data:
            config_kwargs['storage'] = StorageConfig(**data['storage'])
        
        if 'logging' in data:
            config_kwargs['logging'] = LoggingConfig(**data['logging'])
        
        if 'progress' in data:
            config_kwargs['progress'] = ProgressConfig(**data['progress'])
        
        return cls(**config_kwargs)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("config")
        
        # 默认配置文件路径
        self.default_config_path = self.config_dir / "default.yaml"
        self.user_config_path = self.config_dir / "user.yaml"
        
        # 当前加载的配置
        self.current_config: Optional[QuantumVQEFullConfig] = None
    
    def create_default_config(self, force: bool = False) -> Path:
        """创建默认配置文件"""
        if self.default_config_path.exists() and not force:
            self.logger.info("Default config already exists")
            return self.default_config_path
        
        default_config = QuantumVQEFullConfig()
        self.save_config(default_config, self.default_config_path)
        self.logger.info(f"Created default config: {self.default_config_path}")
        return self.default_config_path
    
    def save_config(self, config: QuantumVQEFullConfig, file_path: Union[str, Path]):
        """保存配置到文件"""
        file_path = Path(file_path)
        
        try:
            config_dict = config.to_dict()
            
            # 添加元数据
            config_dict['_metadata'] = {
                'created_at': datetime.now().isoformat(),
                'version': config.project.version,
                'config_file_version': '1.0'
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            self.logger.info(f"Saved config to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
    
    def load_config(self, file_path: Union[str, Path]) -> QuantumVQEFullConfig:
        """从文件加载配置"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 移除元数据
            if '_metadata' in config_dict:
                del config_dict['_metadata']
            
            config = QuantumVQEFullConfig.from_dict(config_dict)
            self.current_config = config
            
            self.logger.info(f"Loaded config from {file_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def load_config_with_override(self, 
                                 base_config_path: Union[str, Path],
                                 override_config_path: Optional[Union[str, Path]] = None,
                                 env_overrides: bool = True) -> QuantumVQEFullConfig:
        """加载配置并应用覆盖"""
        
        # 加载基础配置
        base_config = self.load_config(base_config_path)
        config_dict = base_config.to_dict()
        
        # 应用覆盖配置文件
        if override_config_path and Path(override_config_path).exists():
            with open(override_config_path, 'r', encoding='utf-8') as f:
                override_dict = yaml.safe_load(f)
            
            # 递归合并配置
            config_dict = self._merge_configs(config_dict, override_dict)
            self.logger.info(f"Applied config overrides from {override_config_path}")
        
        # 应用环境变量覆盖
        if env_overrides:
            env_config = self._get_env_overrides()
            if env_config:
                config_dict = self._merge_configs(config_dict, env_config)
                self.logger.info("Applied environment variable overrides")
        
        # 创建最终配置
        final_config = QuantumVQEFullConfig.from_dict(config_dict)
        self.current_config = final_config
        
        return final_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _get_env_overrides(self) -> Dict[str, Any]:
        """从环境变量获取配置覆盖"""
        env_config = {}
        
        # 定义环境变量映射
        env_mapping = {
            'QVQE_NUM_CIRCUITS': ('circuit_generation', 'num_circuits', int),
            'QVQE_NUM_QUBITS': ('circuit_generation', 'num_qubits', int),
            'QVQE_VQE_ENABLED': ('vqe', 'enabled', bool),
            'QVQE_VQE_BATCH_SIZE': ('vqe', 'batch_size', int),
            'QVQE_VQE_MAX_WORKERS': ('vqe', 'max_workers', int),
            'QVQE_VQE_OPTIMIZER': ('vqe', 'optimizer', str),
            'QVQE_EXPR_ENABLED': ('expressibility', 'enabled', bool),
            'QVQE_EXPR_SAMPLES': ('expressibility', 'samples', int),
            'QVQE_NOISY_ENABLED': ('noisy_vqe', 'enabled', bool),
            'QVQE_LOG_LEVEL': ('logging', 'level', str),
            'QVQE_OUTPUT_DIR': ('project', 'base_output_dir', str),
        }
        
        for env_var, (section, key, value_type) in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        parsed_value = int(env_value)
                    else:
                        parsed_value = env_value
                    
                    if section not in env_config:
                        env_config[section] = {}
                    env_config[section][key] = parsed_value
                    
                except ValueError as e:
                    self.logger.warning(f"Invalid environment variable value {env_var}={env_value}: {e}")
        
        return env_config
    
    def get_current_config(self) -> Optional[QuantumVQEFullConfig]:
        """获取当前配置"""
        return self.current_config
    
    def validate_config(self, config: QuantumVQEFullConfig) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 验证电路生成配置
        if config.circuit_generation.num_circuits <= 0:
            errors.append("num_circuits must be positive")
        
        if config.circuit_generation.num_qubits <= 0:
            errors.append("num_qubits must be positive")
        
        # 验证VQE配置
        if config.vqe.enabled:
            if config.vqe.batch_size <= 0:
                errors.append("VQE batch_size must be positive")
            
            if config.vqe.n_repeat <= 0:
                errors.append("VQE n_repeat must be positive")
            
            valid_optimizers = ['L_BFGS_B', 'SPSA', 'COBYLA']
            if config.vqe.optimizer not in valid_optimizers:
                errors.append(f"VQE optimizer must be one of {valid_optimizers}")
        
        # 验证表达性配置
        if config.expressibility.enabled:
            if config.expressibility.samples <= 0:
                errors.append("Expressibility samples must be positive")
            
            if config.expressibility.bins <= 0:
                errors.append("Expressibility bins must be positive")
        
        # 验证日志配置
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_log_levels:
            errors.append(f"Log level must be one of {valid_log_levels}")
        
        return errors
    
    def create_config_template(self, template_name: str) -> Path:
        """创建配置模板"""
        
        if template_name == "quick_test":
            # 快速测试配置
            config = QuantumVQEFullConfig()
            config.circuit_generation.num_circuits = 100
            config.circuit_generation.num_qubits = 4
            config.vqe.batch_size = 20
            config.vqe.n_repeat = 1
            config.expressibility.enabled = False
            config.noisy_vqe.enabled = False
            
        elif template_name == "production":
            # 生产环境配置
            config = QuantumVQEFullConfig()
            config.circuit_generation.num_circuits = 10000
            config.circuit_generation.num_qubits = 8
            config.vqe.batch_size = 100
            config.vqe.n_repeat = 3
            config.expressibility.enabled = True
            config.noisy_vqe.enabled = True
            config.logging.level = "WARNING"
            
        elif template_name == "development":
            # 开发环境配置
            config = QuantumVQEFullConfig()
            config.circuit_generation.num_circuits = 1000
            config.vqe.verbose = True
            config.logging.level = "DEBUG"
            config.progress.show_individual_progress = True
            
        else:
            raise ValueError(f"Unknown template: {template_name}")
        
        template_path = self.config_dir / f"{template_name}.yaml"
        self.save_config(config, template_path)
        return template_path


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def setup_global_config_manager(config_dir: str = "configs") -> ConfigManager:
    """设置全局配置管理器"""
    global _global_config_manager
    _global_config_manager = ConfigManager(config_dir)
    return _global_config_manager


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    if _global_config_manager is None:
        setup_global_config_manager()
    return _global_config_manager


def load_project_config(config_path: str = "configs/default.yaml") -> QuantumVQEFullConfig:
    """加载项目配置的便捷函数"""
    manager = get_config_manager()
    
    # 如果配置文件不存在，创建默认配置
    if not Path(config_path).exists():
        manager.create_default_config()
    
    return manager.load_config(config_path)


if __name__ == "__main__":
    # 测试配置管理器
    from ..utils.logger import setup_global_logger
    
    # 设置日志
    setup_global_logger({'name': 'test_config', 'level': 'INFO'})
    
    # 创建配置管理器
    manager = ConfigManager(config_dir="test_configs")
    
    # 创建默认配置
    default_path = manager.create_default_config(force=True)
    print(f"创建默认配置: {default_path}")
    
    # 创建模板配置
    quick_test_path = manager.create_config_template("quick_test")
    print(f"创建快速测试配置: {quick_test_path}")
    
    # 加载和验证配置
    config = manager.load_config(quick_test_path)
    errors = manager.validate_config(config)
    
    if errors:
        print(f"配置验证错误: {errors}")
    else:
        print("配置验证通过")
    
    # 显示配置信息
    print(f"电路数量: {config.circuit_generation.num_circuits}")
    print(f"量子比特数: {config.circuit_generation.num_qubits}")
    print(f"VQE启用: {config.vqe.enabled}")
    print(f"表达性启用: {config.expressibility.enabled}")
    
    print("配置管理器测试完成")
