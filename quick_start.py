#!/usr/bin/env python3
# quick_start.py - 快速开始脚本

"""
量子VQE项目 - 快速开始
===================

这个脚本会创建一个最小化的可工作版本，让您立即开始使用新系统。

运行此脚本后，您可以：
1. 测试新的配置系统
2. 运行小规模仿真
3. 体验新的工作流程
"""

import os
import sys
from pathlib import Path


# 最小化的TFIM模型实现
TFIM_CODE = '''
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver


class TFIM:
    def __init__(self, size=6, J=1, g=1):
        """
        初始化TFIM类的实例。

        参数:
        size (int): 系统的大小。
        J (float): 相互作用强度。
        g (float): 横场强度。
        """
        self.size = size
        self.J = J
        self.g = g
        self.H_list = []

    def generate_hamiltonian(self):
        """
        生成TFIM系统的哈密顿量。

        返回:
        SparsePauliOp: 系统的哈密顿量。
        """
        self.H_list = []
        for i in range(self.size):
            term = ''.join('Z' if k == i or k == (i + 1) %
                           self.size else 'I' for k in range(self.size))
            self.H_list.append((term, -self.J))
        for i in range(self.size):
            term = ''.join('X' if k == i else 'I' for k in range(self.size))
            self.H_list.append((term, -self.J * self.g))
        self.Hamiltonian = SparsePauliOp.from_list(self.H_list)
        return self.Hamiltonian

    def compute_energy(self):
        """
        计算TFIM系统的基态能量。

        返回:
        float: 系统的基态能量。
        """
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(self.Hamiltonian)
        return result.eigenvalue.real

    def get_hamiltonian_and_energy(self):
        """
        获取TFIM系统的哈密顿量和基态能量。

        返回:
        tuple: 包含哈密顿量 (SparsePauliOp) 和基态能量 (float) 的元组。
        """
        hamiltonian = self.generate_hamiltonian()
        energy = self.compute_energy()
        return hamiltonian, energy
'''

# 简化的电路生成器
GENERATOR_CODE = '''
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class SimpleCircuitGenerator:
    """简化的量子电路生成器 - 用于快速测试"""
    
    def __init__(self):
        self.logger = None
    
    def generate_tfim_quantum_circuits(self, 
                                      num_circuits=10, 
                                      num_qubits=4, 
                                      max_gates=20,
                                      max_add_count=20,
                                      gate_stddev=1.0,
                                      gate_bias=0.5):
        """生成简单的参数化量子电路"""
        circuits = []
        
        for i in range(num_circuits):
            qc = QuantumCircuit(num_qubits)
            param_count = 0
            
            # 添加H门层
            for q in range(num_qubits):
                qc.h(q)
            
            # 添加参数化层
            depth = np.random.randint(2, 5)
            for layer in range(depth):
                # 单量子比特旋转门
                for q in range(num_qubits):
                    gate_type = np.random.choice(['rx', 'ry', 'rz'])
                    param = Parameter(f"theta_{param_count}")
                    getattr(qc, gate_type)(param, q)
                    param_count += 1
                
                # 纠缠层
                for q in range(num_qubits - 1):
                    qc.cx(q, q + 1)
            
            qc.metadata = {"id": i}
            circuits.append(qc)
        
        return circuits
'''

# 测试配置
TEST_CONFIG = '''
# 快速测试配置
project:
  name: "quick_test_quantum_vqe"
  version: "1.0.0"
  base_output_dir: "test_output"
  timestamp_dirs: true

circuit_generation:
  num_circuits: 20
  num_qubits: 4
  max_depth: 10
  max_gates: 20
  tfim_J: 1.0
  tfim_g: 1.0
  generator_type: "simple"

vqe:
  enabled: true
  batch_size: 5
  max_workers: 2
  n_repeat: 1
  optimizer: "L_BFGS_B"
  verbose: false
  suppress_optimizer_output: true

expressibility:
  enabled: true
  samples: 1000
  bins: 30
  batch_size: 5

noisy_vqe:
  enabled: false

storage:
  base_dir: "data"
  compression: false
  backup_enabled: false

logging:
  level: "INFO"
  format_type: "human"
  enable_console: true
  log_dir: "logs"

progress:
  show_overall_progress: true
  show_individual_progress: true
'''

# 简化的运行脚本
RUN_SCRIPT = '''#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# 添加src目录到路径
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

def main():
    """运行快速测试"""
    print("量子VQE快速测试")
    print("="*30)
    
    try:
        # 导入必要模块
        from src.models.tfim import TFIM
        from src.generators.simple_generator import SimpleCircuitGenerator
        
        # 1. 创建TFIM模型
        print("1. 创建TFIM模型...")
        tfim = TFIM(size=4, J=1.0, g=1.0)
        hamiltonian, exact_energy = tfim.get_hamiltonian_and_energy()
        print(f"   精确基态能量: {exact_energy:.6f}")
        
        # 2. 生成测试电路
        print("2. 生成测试电路...")
        generator = SimpleCircuitGenerator()
        circuits = generator.generate_tfim_quantum_circuits(num_circuits=5, num_qubits=4)
        print(f"   生成了 {len(circuits)} 个电路")
        
        # 3. 简单VQE测试
        print("3. 运行简单VQE测试...")
        from qiskit.primitives import Estimator
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import L_BFGS_B
        
        estimator = Estimator()
        optimizer = L_BFGS_B()
        
        # 测试第一个电路
        test_circuit = circuits[0]
        vqe = VQE(estimator, test_circuit, optimizer)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        print(f"   VQE结果: {result.eigenvalue.real:.6f}")
        print(f"   与精确值的差异: {abs(result.eigenvalue.real - exact_energy):.6f}")
        
        print("\\n✓ 快速测试完成！系统运行正常。")
        print("\\n下一步:")
        print("- 安装完整的新框架文件")
        print("- 运行完整的流水线测试")
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装必要的依赖: pip install qiskit qiskit-algorithms")
    except Exception as e:
        print(f"✗ 运行错误: {e}")

if __name__ == "__main__":
    main()
'''


def create_minimal_project():
    """创建最小化项目结构"""
    
    # 创建目录
    dirs = [
        "src",
        "src/models", 
        "src/generators",
        "configs",
        "test_scripts"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    # 创建 __init__.py 文件
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/generators/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    # 创建核心文件
    files_to_create = {
        "src/models/tfim.py": TFIM_CODE,
        "src/generators/simple_generator.py": GENERATOR_CODE,
        "configs/quick_test.yaml": TEST_CONFIG,
        "test_scripts/quick_test.py": RUN_SCRIPT
    }
    
    for file_path, content in files_to_create.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✓ 创建文件: {file_path}")
    
    # 使测试脚本可执行
    os.chmod("test_scripts/quick_test.py", 0o755)


def main():
    """主函数"""
    print("量子VQE项目 - 快速开始设置")
    print("="*35)
    
    print("\\n这个脚本会创建一个最小化的可工作版本，让您立即开始测试。")
    
    choice = input("\\n是否继续创建快速开始项目？(y/N): ").strip().lower()
    
    if choice not in ('y', 'yes'):
        print("取消设置。")
        return
    
    print("\\n创建最小化项目结构...")
    create_minimal_project()
    
    print("\\n" + "="*50)
    print("快速开始项目创建完成！")
    print("="*50)
    
    print("\\n📋 现在您可以:")
    print("1. 安装基本依赖:")
    print("   pip install qiskit qiskit-algorithms numpy")
    
    print("\\n2. 运行快速测试:")
    print("   python test_scripts/quick_test.py")
    
    print("\\n3. 如果测试成功，继续安装完整框架:")
    print("   - 下载我提供的所有框架文件")
    print("   - 运行完整的项目初始化")
    
    print("\\n这个最小版本包含:")
    print("   ✓ 基本的TFIM模型")
    print("   ✓ 简单的电路生成器") 
    print("   ✓ VQE测试脚本")
    print("   ✓ 快速测试配置")
    
    print("\\n🚀 现在可以开始体验新的工作流程了！")


if __name__ == "__main__":
    main()
