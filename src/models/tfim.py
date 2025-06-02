
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
