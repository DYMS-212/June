
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
