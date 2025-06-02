#!/usr/bin/env python3
# scripts/run_simulation.py
"""
Quantum VQE Simulation - Main Execution Script
==============================================

This script provides a command-line interface to run the quantum VQE simulation pipeline.
It supports configuration files, command-line overrides, and selective execution of different
simulation types.

Usage Examples:
    # Run with default configuration
    python run_simulation.py
    
    # Run with custom configuration
    python run_simulation.py --config configs/production.yaml
    
    # Run only VQE simulation
    python run_simulation.py --vqe-only
    
    # Use existing circuits
    python run_simulation.py --circuits circuits_20241202_143022.pkl
    
    # Override configuration parameters
    python run_simulation.py --num-circuits 1000 --num-qubits 6
    
    # Generate quick test configuration
    python run_simulation.py --create-config quick_test
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from src.pipeline import QuantumVQEPipeline, run_quantum_vqe_pipeline
from src.utils.config import ConfigManager, QuantumVQEFullConfig
from src.utils.logger import setup_global_logger, get_logger


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    
    parser = argparse.ArgumentParser(
        description="Quantum VQE Circuit Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default config
  %(prog)s --config production.yaml          # Run with custom config
  %(prog)s --vqe-only --num-circuits 500     # VQE only with 500 circuits
  %(prog)s --circuits existing.pkl           # Use existing circuits
  %(prog)s --create-config quick_test        # Create test config template
        """
    )
    
    # 配置相关参数
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )
    config_group.add_argument(
        '--create-config',
        type=str,
        choices=['default', 'quick_test', 'production', 'development'],
        help='Create configuration template and exit'
    )
    config_group.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    # 执行控制参数
    execution_group = parser.add_argument_group('Execution Control')
    execution_group.add_argument(
        '--circuits',
        type=str,
        help='Use existing circuit file instead of generating new ones'
    )
    execution_group.add_argument(
        '--vqe-only',
        action='store_true',
        help='Run only VQE simulation'
    )
    execution_group.add_argument(
        '--expressibility-only',
        action='store_true',
        help='Run only expressibility calculation'
    )
    execution_group.add_argument(
        '--noisy-vqe-only',
        action='store_true',
        help='Run only noisy VQE simulation'
    )
    execution_group.add_argument(
        '--no-generate',
        action='store_true',
        help='Skip circuit generation (requires --circuits)'
    )
    
    # 电路生成参数覆盖
    circuit_group = parser.add_argument_group('Circuit Generation Overrides')
    circuit_group.add_argument(
        '--num-circuits',
        type=int,
        help='Number of circuits to generate'
    )
    circuit_group.add_argument(
        '--num-qubits',
        type=int,
        help='Number of qubits per circuit'
    )
    circuit_group.add_argument(
        '--max-depth',
        type=int,
        help='Maximum circuit depth'
    )
    circuit_group.add_argument(
        '--max-gates',
        type=int,
        help='Maximum number of gates'
    )
    
    # VQE参数覆盖
    vqe_group = parser.add_argument_group('VQE Simulation Overrides')
    vqe_group.add_argument(
        '--vqe-batch-size',
        type=int,
        help='VQE batch size'
    )
    vqe_group.add_argument(
        '--vqe-max-workers',
        type=int,
        help='VQE maximum worker processes'
    )
    vqe_group.add_argument(
        '--vqe-n-repeat',
        type=int,
        help='Number of VQE repetitions per circuit'
    )
    vqe_group.add_argument(
        '--vqe-optimizer',
        type=str,
        choices=['L_BFGS_B', 'SPSA', 'COBYLA'],
        help='VQE optimizer'
    )
    vqe_group.add_argument(
        '--vqe-verbose',
        action='store_true',
        help='Enable VQE verbose output'
    )
    
    # 表达性参数覆盖
    expr_group = parser.add_argument_group('Expressibility Overrides')
    expr_group.add_argument(
        '--expr-samples',
        type=int,
        help='Expressibility sampling count'
    )
    expr_group.add_argument(
        '--expr-bins',
        type=int,
        help='Expressibility histogram bins'
    )
    
    # 输出和日志参数
    output_group = parser.add_argument_group('Output and Logging')
    output_group.add_argument(
        '--output-dir',
        type=str,
        help='Output directory'
    )
    output_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    output_group.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Disable timestamp in output directory'
    )
    output_group.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal console output'
    )
    
    # 其他选项
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    other_group.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )
    
    return parser


def apply_cli_overrides(config: QuantumVQEFullConfig, args: argparse.Namespace) -> QuantumVQEFullConfig:
    """应用命令行参数覆盖到配置"""
    
    # 电路生成参数
    if args.num_circuits is not None:
        config.circuit_generation.num_circuits = args.num_circuits
    if args.num_qubits is not None:
        config.circuit_generation.num_qubits = args.num_qubits
    if args.max_depth is not None:
        config.circuit_generation.max_depth = args.max_depth
    if args.max_gates is not None:
        config.circuit_generation.max_gates = args.max_gates
    
    # VQE参数
    if args.vqe_batch_size is not None:
        config.vqe.batch_size = args.vqe_batch_size
    if args.vqe_max_workers is not None:
        config.vqe.max_workers = args.vqe_max_workers
    if args.vqe_n_repeat is not None:
        config.vqe.n_repeat = args.vqe_n_repeat
    if args.vqe_optimizer is not None:
        config.vqe.optimizer = args.vqe_optimizer
    if args.vqe_verbose:
        config.vqe.verbose = True
    
    # 表达性参数
    if args.expr_samples is not None:
        config.expressibility.samples = args.expr_samples
    if args.expr_bins is not None:
        config.expressibility.bins = args.expr_bins
    
    # 执行控制
    if args.vqe_only:
        config.vqe.enabled = True
        config.expressibility.enabled = False
        config.noisy_vqe.enabled = False
    elif args.expressibility_only:
        config.vqe.enabled = False
        config.expressibility.enabled = True
        config.noisy_vqe.enabled = False
    elif args.noisy_vqe_only:
        config.vqe.enabled = False
        config.expressibility.enabled = False
        config.noisy_vqe.enabled = True
    
    # 输出和日志
    if args.output_dir is not None:
        config.project.base_output_dir = args.output_dir
    if args.log_level is not None:
        config.logging.level = args.log_level
    if args.no_timestamp:
        config.project.timestamp_dirs = False
    if args.quiet:
        config.logging.enable_console = False
        config.progress.show_individual_progress = False
    
    return config


def create_config_template(template_name: str, config_dir: str = "configs") -> None:
    """创建配置模板"""
    config_manager = ConfigManager(config_dir)
    
    try:
        template_path = config_manager.create_config_template(template_name)
        print(f"✓ Created configuration template: {template_path}")
        print(f"  You can now edit this file and run:")
        print(f"  python run_simulation.py --config {template_path}")
    except Exception as e:
        print(f"✗ Failed to create configuration template: {e}")
        sys.exit(1)


def validate_configuration(config_path: str) -> None:
    """验证配置文件"""
    config_manager = ConfigManager()
    
    try:
        config = config_manager.load_config(config_path)
        errors = config_manager.validate_config(config)
        
        if errors:
            print("✗ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("✓ Configuration validation passed")
            print(f"  Config file: {config_path}")
            print(f"  Project: {config.project.name} v{config.project.version}")
            print(f"  Circuits: {config.circuit_generation.num_circuits}")
            print(f"  Qubits: {config.circuit_generation.num_qubits}")
            print(f"  VQE enabled: {config.vqe.enabled}")
            print(f"  Expressibility enabled: {config.expressibility.enabled}")
            print(f"  Noisy VQE enabled: {config.noisy_vqe.enabled}")
            
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        sys.exit(1)


def show_dry_run_info(config: QuantumVQEFullConfig, existing_circuits: Optional[str] = None) -> None:
    """显示干运行信息"""
    print("=" * 60)
    print("DRY RUN - What would be executed:")
    print("=" * 60)
    
    print(f"Project: {config.project.name} v{config.project.version}")
    print(f"Output Directory: {config.project.base_output_dir}")
    print(f"Timestamp Dirs: {config.project.timestamp_dirs}")
    print()
    
    if existing_circuits:
        print(f"Circuit Loading:")
        print(f"  Load from: {existing_circuits}")
    else:
        print(f"Circuit Generation:")
        print(f"  Number of circuits: {config.circuit_generation.num_circuits}")
        print(f"  Number of qubits: {config.circuit_generation.num_qubits}")
        print(f"  Max depth: {config.circuit_generation.max_depth}")
        print(f"  Max gates: {config.circuit_generation.max_gates}")
    print()
    
    print("Simulations to run:")
    if config.vqe.enabled:
        print(f"  ✓ VQE (batch_size={config.vqe.batch_size}, n_repeat={config.vqe.n_repeat})")
    else:
        print("  ✗ VQE (disabled)")
    
    if config.expressibility.enabled:
        print(f"  ✓ Expressibility (samples={config.expressibility.samples})")
    else:
        print("  ✗ Expressibility (disabled)")
    
    if config.noisy_vqe.enabled:
        print(f"  ✓ Noisy VQE (batch_size={config.noisy_vqe.batch_size})")
    else:
        print("  ✗ Noisy VQE (disabled)")
    
    print()
    print("=" * 60)


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 处理特殊命令
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    if args.validate_config:
        validate_configuration(args.config)
        return
    
    # 加载和验证配置
    try:
        config_manager = ConfigManager()
        
        # 如果配置文件不存在，创建默认配置
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file {config_path} not found, creating default...")
            config_manager.create_default_config()
        
        # 加载配置
        config = config_manager.load_config(args.config)
        
        # 应用命令行覆盖
        config = apply_cli_overrides(config, args)
        
        # 验证最终配置
        errors = config_manager.validate_config(config)
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # 检查参数约束
    if args.no_generate and not args.circuits:
        print("Error: --no-generate requires --circuits")
        sys.exit(1)
    
    # 显示干运行信息
    if args.dry_run:
        show_dry_run_info(config, args.circuits)
        return
    
    # 创建并运行流水线
    try:
        print("=" * 60)
        print(f"Starting Quantum VQE Simulation Pipeline")
        print("=" * 60)
        print(f"Project: {config.project.name} v{config.project.version}")
        print(f"Configuration: {args.config}")
        
        if args.circuits:
            print(f"Using existing circuits: {args.circuits}")
        else:
            print(f"Generating {config.circuit_generation.num_circuits} circuits with {config.circuit_generation.num_qubits} qubits")
        
        enabled_sims = []
        if config.vqe.enabled:
            enabled_sims.append("VQE")
        if config.expressibility.enabled:
            enabled_sims.append("Expressibility")
        if config.noisy_vqe.enabled:
            enabled_sims.append("Noisy VQE")
        
        print(f"Enabled simulations: {', '.join(enabled_sims) if enabled_sims else 'None'}")
        print("=" * 60)
        
        # 运行流水线
        pipeline = QuantumVQEPipeline(config)
        
        try:
            results = pipeline.run_full_pipeline(args.circuits)
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)
            print(f"Results saved to: {results['pipeline_info']['output_directory']}")
            print(f"Execution time: {results['pipeline_info']['execution_time']:.2f} seconds")
            
            # 显示结果摘要
            circuit_info = results['circuit_info']
            print(f"\nCircuit Information:")
            print(f"  Total circuits: {circuit_info['total_circuits']}")
            print(f"  Qubits per circuit: {circuit_info['num_qubits']}")
            print(f"  Exact ground energy: {circuit_info['exact_ground_energy']:.6f}")
            
            sim_results = results['simulation_results']
            if sim_results['vqe'].get('status') != 'disabled':
                vqe_res = sim_results['vqe']
                print(f"\nVQE Results:")
                print(f"  Success rate: {vqe_res.get('success_count', 0)}/{vqe_res.get('total_circuits', 0)} "
                      f"({vqe_res.get('success_rate', 0)*100:.1f}%)")
                if 'energy_statistics' in vqe_res and 'min_energy' in vqe_res['energy_statistics']:
                    energy_stats = vqe_res['energy_statistics']
                    print(f"  Best energy found: {energy_stats['min_energy']:.6f}")
                    print(f"  Average energy: {energy_stats['mean_energy']:.6f}")
            
            print("=" * 60)
            
        finally:
            pipeline.cleanup()
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
