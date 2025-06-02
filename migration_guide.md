# 量子VQE项目 - 完整迁移指南

## 📁 第一步：创建完整项目结构

请按以下结构组织您的文件：

```
quantum_vqe_project/
├── src/
│   ├── __init__.py                    # 新建（空文件）
│   ├── models/
│   │   ├── __init__.py               # 新建（空文件）
│   │   └── tfim.py                   # 从您的 TFIM.py 复制并重命名
│   ├── generators/
│   │   ├── __init__.py               # 新建（空文件）
│   │   └── layerwise.py              # 从您的 layerwise.py 复制
│   ├── simulators/
│   │   ├── __init__.py               # 新建（空文件）
│   │   ├── base_simulator.py         # 我提供的新文件
│   │   ├── vqe_simulator.py          # 我提供的新文件
│   │   ├── expressibility_simulator.py # 我提供的新文件
│   │   └── noisy_vqe_simulator.py    # 需要基于您的 noise_vqe.py 创建
│   ├── utils/
│   │   ├── __init__.py               # 新建（空文件）
│   │   ├── logger.py                 # 我提供的新文件
│   │   ├── storage.py                # 我提供的新文件
│   │   ├── progress.py               # 我提供的新文件
│   │   ├── config.py                 # 我提供的新文件
│   │   └── expressibility.py        # 从您的 expressibility.py 复制
│   └── pipeline.py                   # 我提供的新文件
├── configs/
│   └── default.yaml                  # 我提供的新文件
├── scripts/
│   ├── run_simulation.py             # 我提供的新文件
│   └── analyze_results.py            # 从您的 analyze_results.py 适配
├── requirements.txt                   # 新建
└── README.md                         # 新建
```

## 📋 第二步：需要您操作的文件

### 1. 复制您的原始文件到新位置

```bash
# 在您的项目根目录执行
mkdir -p src/models src/generators src/simulators src/utils configs scripts

# 复制核心文件
cp TFIM.py src/models/tfim.py
cp layerwise.py src/generators/layerwise.py
cp expressibility.py src/utils/expressibility.py
cp analyze_results.py scripts/analyze_results.py

# 创建空的 __init__.py 文件
touch src/__init__.py
touch src/models/__init__.py
touch src/generators/__init__.py
touch src/simulators/__init__.py
touch src/utils/__init__.py
```

### 2. 创建 requirements.txt

```txt
# requirements.txt
qiskit>=0.44.0
qiskit-algorithms>=0.2.0
qiskit-aer>=0.12.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
PyYAML>=6.0
h5py>=3.7.0
pandas>=1.5.0
```

### 3. 修改导入路径

您需要在一些文件中更新导入路径：

#### `src/models/tfim.py` (从您的 TFIM.py 复制后修改)
```python
# 保持原有内容，只需要确保类名是 TFIM
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver

class TFIM:
    # ... 您的原有实现 ...
```

#### `src/generators/layerwise.py` (从您的 layerwise.py 复制后修改)
```python
# 在文件开头添加
import sys
from pathlib import Path

# 确保可以导入 utils
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 您的原有代码...
```

## 📝 第三步：我为您创建的补充文件

让我为您创建几个关键的缺失文件：
