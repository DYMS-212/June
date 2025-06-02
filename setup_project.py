#!/usr/bin/env python3
# setup_project.py - 项目初始化脚本

"""
量子VQE项目初始化脚本
====================

这个脚本帮助您将现有的代码文件整合到新的项目结构中。

使用方法:
    python setup_project.py

脚本会：
1. 创建新的项目目录结构
2. 指导您复制现有文件
3. 创建必要的配置文件
4. 验证项目设置
"""

import os
import shutil
from pathlib import Path


def create_directory_structure():
    """创建项目目录结构"""
    dirs = [
        "src",
        "src/models",
        "src/generators", 
        "src/simulators",
        "src/utils",
        "configs",
        "scripts",
        "data",
        "data/circuits",
        "data/results", 
        "data/metadata",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    # 创建 __init__.py 文件
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/generators/__init__.py",
        "src/simulators/__init__.py", 
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ 创建文件: {init_file}")


def create_requirements_txt():
    """创建 requirements.txt"""
    requirements = """# Quantum VQE Simulation Requirements
qiskit>=0.44.0
qiskit-algorithms>=0.2.0
qiskit-aer>=0.12.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
PyYAML>=6.0
h5py>=3.7.0
pandas>=1.5.0
scipy>=1.9.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ 创建 requirements.txt")


def create_readme():
    """创建 README.md"""
    readme = """# 量子VQE仿真项目

这是一个重构后的量子VQE（变分量子本征求解器）仿真项目，支持TFIM模型、表达性计算和含噪声仿真。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行快速测试：
   ```bash
   python scripts/run_simulation.py --create-config quick_test
   python scripts/run_simulation.py --config configs/quick_test.yaml
   ```

## 项目结构

- `src/`: 核心源代码
- `configs/`: 配置文件
- `scripts/`: 执行脚本
- `data/`: 数据存储
- `logs/`: 日志文件

## 更多信息

详细使用说明请参考项目文档。
"""
    
    with open("README.md", "w") as f:
        f.write(readme)
    print("✓ 创建 README.md")


def check_existing_files():
    """检查当前目录下的原始文件"""
    important_files = {
        "TFIM.py": "src/models/tfim.py",
        "layerwise.py": "src/generators/layerwise.py", 
        "expressibility.py": "src/utils/expressibility.py",
        "noise_vqe.py": "参考文件 - 已集成到新框架",
        "analyze_results.py": "scripts/analyze_results.py"
    }
    
    print("\n" + "="*50)
    print("检查原始文件...")
    print("="*50)
    
    found_files = []
    missing_files = []
    
    for original_file, target_location in important_files.items():
        if Path(original_file).exists():
            found_files.append((original_file, target_location))
            print(f"✓ 找到: {original_file} -> 应复制到: {target_location}")
        else:
            missing_files.append(original_file)
            print(f"✗ 未找到: {original_file}")
    
    return found_files, missing_files


def copy_file_with_backup(source, target):
    """复制文件，如果目标存在则备份"""
    target_path = Path(target)
    if target_path.exists():
        backup_path = target_path.with_suffix(target_path.suffix + ".backup")
        shutil.copy2(target_path, backup_path)
        print(f"  备份现有文件: {target} -> {backup_path}")
    
    shutil.copy2(source, target)
    print(f"  复制: {source} -> {target}")


def interactive_file_migration(found_files):
    """交互式文件迁移"""
    if not found_files:
        print("\n没有找到需要迁移的文件。")
        return
    
    print(f"\n找到 {len(found_files)} 个文件需要迁移。")
    choice = input("是否现在复制这些文件？(y/N): ").strip().lower()
    
    if choice in ('y', 'yes'):
        print("\n开始复制文件...")
        for source, target in found_files:
            try:
                copy_file_with_backup(source, target)
                print(f"✓ 成功复制: {source}")
            except Exception as e:
                print(f"✗ 复制失败 {source}: {e}")
    else:
        print("\n跳过文件复制。您可以稍后手动复制：")
        for source, target in found_files:
            print(f"  cp {source} {target}")


def show_next_steps():
    """显示后续步骤"""
    print("\n" + "="*50)
    print("项目初始化完成！")
    print("="*50)
    
    print("\n📋 后续步骤：")
    print("1. 安装依赖:")
    print("   pip install -r requirements.txt")
    
    print("\n2. 从我提供的文件中下载以下核心文件到对应位置:")
    core_files = [
        "src/utils/logger.py",
        "src/utils/storage.py", 
        "src/utils/progress.py",
        "src/utils/config.py",
        "src/simulators/base_simulator.py",
        "src/simulators/vqe_simulator.py",
        "src/simulators/expressibility_simulator.py",
        "src/simulators/noisy_vqe_simulator.py",
        "src/pipeline.py",
        "scripts/run_simulation.py",
        "configs/default.yaml"
    ]
    
    for file_path in core_files:
        print(f"   - {file_path}")
    
    print("\n3. 创建并测试配置:")
    print("   python scripts/run_simulation.py --create-config quick_test")
    
    print("\n4. 运行测试:")
    print("   python scripts/run_simulation.py --config configs/quick_test.yaml")
    
    print("\n5. 如果遇到导入错误，请检查文件路径和依赖关系。")
    
    print("\n📂 项目结构已创建完成，您现在可以开始使用新的工作流程了！")


def main():
    """主函数"""
    print("量子VQE项目初始化脚本")
    print("="*30)
    
    print("\n1. 创建项目目录结构...")
    create_directory_structure()
    
    print("\n2. 创建项目文件...")
    create_requirements_txt()
    create_readme()
    
    print("\n3. 检查现有文件...")
    found_files, missing_files = check_existing_files()
    
    if missing_files:
        print(f"\n⚠️  以下文件未找到（如果您没有这些文件，可以跳过）:")
        for file in missing_files:
            print(f"   - {file}")
    
    print("\n4. 文件迁移...")
    interactive_file_migration(found_files)
    
    show_next_steps()


if __name__ == "__main__":
    main()
