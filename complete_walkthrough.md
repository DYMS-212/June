# 量子VQE项目 - 完整使用示例

## 🎯 目标
从零开始，完整体验新的量子VQE仿真流程，包括：
- 项目设置和配置
- 生成量子电路
- 运行VQE仿真
- 计算表达性
- 分析结果

## 📁 第一步：项目设置

### 1.1 创建项目目录结构
```bash
# 创建主项目目录
mkdir quantum_vqe_demo
cd quantum_vqe_demo

# 创建子目录
mkdir -p src/{models,generators,simulators,utils}
mkdir -p configs scripts
mkdir -p data/{circuits,results,metadata}
mkdir -p logs

# 创建__init__.py文件
touch src/__init__.py
touch src/models/__init__.py
touch src/generators/__init__.py
touch src/simulators/__init__.py
touch src/utils/__init__.py
```

### 1.2 放置所有代码文件
按照我之前提供的文件清单，将所有artifacts下载并保存到正确位置：

```
quantum_vqe_demo/
├── src/
│   ├── models/
│   │   └── tfim.py              # 您的原TFIM.py
│   ├── generators/
│   │   └── layerwise.py         # 您的原layerwise.py
│   ├── simulators/
│   │   ├── base_simulator.py    # 我的artifact: simulator_framework
│   │   ├── vqe_simulator.py     # 我的artifact: vqe_simulator
│   │   ├── expressibility_simulator.py # 我的artifact: expressibility_simulator
│   │   └── noisy_vqe_simulator.py # 我的artifact: noisy_vqe_adapter
│   ├── utils/
│   │   ├── logger.py            # 我的artifact: logger_system
│   │   ├── storage.py           # 我的artifact: storage_system
│   │   ├── progress.py          # 我的artifact: progress_system
│   │   ├── config.py            # 我的artifact: config_management
│   │   └── expressibility.py   # 您的原expressibility.py
│   └── pipeline.py              # 我的artifact: main_pipeline
├── configs/
│   └── default.yaml             # 我的artifact: default_config_yaml
├── scripts/
│   ├── run_simulation.py        # 我的artifact: main_script
│   └── analyze_results.py       # 您的原analyze_results.py
└── requirements.txt
```

### 1.3 安装依赖
```bash
# 创建requirements.txt
cat > requirements.txt << EOF
qiskit>=0.44.0
qiskit-algorithms>=0.2.0
qiskit-aer>=0.12.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
PyYAML>=6.0
pandas>=1.5.0
EOF

# 安装依赖
pip install -r requirements.txt
```

## 🚀 第二步：创建和验证配置

### 2.1 创建快速测试配置
```bash
python scripts/run_simulation.py --create-config quick_test
```

这会创建 `configs/quick_test.yaml`，内容类似：
```yaml
project:
  name: "quick_test_quantum_vqe"
  base_output_dir: "output"
  timestamp_dirs: true

circuit_generation:
  num_circuits: 100      # 少量电路用于测试
  num_qubits: 4          # 小规模量子比特
  max_depth: 20
  max_gates: 20

vqe:
  enabled: true
  batch_size: 20
  n_repeat: 1
  optimizer: "L_BFGS_B"

expressibility:
  enabled: true
  samples: 1000          # 较少样本用于快速测试
  bins: 50

noisy_vqe:
  enabled: false         # 先不启用噪声仿真
```

### 2.2 验证配置
```bash
python scripts/run_simulation.py --validate-config --config configs/quick_test.yaml
```

应该看到：
```
✓ Configuration validation passed
  Config file: configs/quick_test.yaml
  Project: quick_test_quantum_vqe v1.0.0
  Circuits: 100
  Qubits: 4
  VQE enabled: True
  Expressibility enabled: True
  Noisy VQE enabled: False
```

## 🔧 第三步：运行快速测试

### 3.1 查看执行计划（干运行）
```bash
python scripts/run_simulation.py --config configs/quick_test.yaml --dry-run
```

输出示例：
```
============================================================
DRY RUN - What would be executed:
============================================================
Project: quick_test_quantum_vqe v1.0.0
Output Directory: output
Timestamp Dirs: True

Circuit Generation:
  Number of circuits: 100
  Number of qubits: 4
  Max depth: 20
  Max gates: 20

Simulations to run:
  ✓ VQE (batch_size=20, n_repeat=1)
  ✓ Expressibility (samples=1000)
  ✗ Noisy VQE (disabled)
============================================================
```

### 3.2 运行完整仿真
```bash
python scripts/run_simulation.py --config configs/quick_test.yaml
```

您会看到实时进度显示：
```
============================================================
Starting Quantum VQE Simulation Pipeline
============================================================
Project: quick_test_quantum_vqe v1.0.0
Configuration: configs/quick_test.yaml
Generating 100 circuits with 4 qubits
Enabled simulations: VQE, Expressibility
============================================================

Generating quantum circuits: 100%|████████| 100/100 [00:02<00:00, 45.2item/s]
Running VQE simulation: 100%|████████| 100/100 [01:23<00:00, 1.20item/s]
Calculating circuit expressibility: 100%|████████| 100/100 [00:45<00:00, 2.22item/s]

============================================================
Pipeline completed successfully!
============================================================
Results saved to: output/quick_test_quantum_vqe_20241202_143022
Execution time: 152.34 seconds

Circuit Information:
  Total circuits: 100
  Qubits per circuit: 4
  Exact ground energy: -6.000000

VQE Results:
  Success rate: 98/100 (98.0%)
  Best energy found: -5.998234
  Average energy: -5.945123
============================================================
```

## 📊 第四步：查看和分析结果

### 4.1 检查输出目录结构
```bash
ls -la output/quick_test_quantum_vqe_20241202_143022/
```

输出：
```
drwxr-xr-x data/
drwxr-xr-x logs/
-rw-r--r-- pipeline_summary.txt

ls -la output/quick_test_quantum_vqe_20241202_143022/data/
```

输出：
```
drwxr-xr-x circuits/
drwxr-xr-x results/
drwxr-xr-x metadata/

circuits/
├── circuits_20241202_143022.pkl

results/
├── vqe_batch_1_results.json
├── vqe_batch_2_results.json
├── ...
├── expressibility_batch_1_results.json
├── ...
├── vqe_final_results_20241202_143500.json
├── expressibility_final_results_20241202_143600.json
└── pipeline_results_20241202_143700.json
```

### 4.2 查看执行摘要
```bash
cat output/quick_test_quantum_vqe_20241202_143022/pipeline_summary.txt
```

输出示例：
```
============================================================
Pipeline Execution Summary: quick_test_quantum_vqe
============================================================

Execution Time: 152.34 seconds
Output Directory: output/quick_test_quantum_vqe_20241202_143022
Timestamp: 2024-12-02T14:37:00.123456

Circuit Information:
  Total Circuits: 100
  Qubits per Circuit: 4
  Exact Ground Energy: -6.000000

Simulation Results:
  VQE Simulation:
    Success Rate: 98/100 (98.0%)
    Simulation Time: 83.45s
    Best Energy: -5.998234
    Average Energy: -5.945123
    Energy Std: 0.023456
  Expressibility: Completed
  Noisy VQE: Disabled
============================================================
```

### 4.3 分析具体结果文件
```bash
# 查看最终汇总结果（JSON格式）
python -c "
import json
with open('output/quick_test_quantum_vqe_20241202_143022/data/results/pipeline_results_20241202_143700.json') as f:
    data = json.load(f)
    
print('=== Pipeline Info ===')
print(f'Execution time: {data[\"pipeline_info\"][\"execution_time\"]:.2f}s')
print(f'Total circuits: {data[\"circuit_info\"][\"total_circuits\"]}')
print(f'Exact energy: {data[\"circuit_info\"][\"exact_ground_energy\"]:.6f}')

print('\n=== VQE Results ===')
vqe = data['simulation_results']['vqe']
print(f'Success rate: {vqe[\"success_count\"]}/{vqe[\"total_circuits\"]} ({vqe[\"success_rate\"]*100:.1f}%)')
if 'energy_statistics' in vqe:
    stats = vqe['energy_statistics']
    print(f'Best energy: {stats[\"min_energy\"]:.6f}')
    print(f'Average energy: {stats[\"mean_energy\"]:.6f}')
    print(f'Energy std: {stats[\"std_energy\"]:.6f}')

print('\n=== Expressibility Results ===')
expr = data['simulation_results']['expressibility']
if expr.get('status') != 'disabled':
    print(f'Status: {expr[\"status\"]}')
    if 'expressibility_statistics' in expr:
        stats = expr['expressibility_statistics']
        print(f'Mean expressibility: {stats[\"mean_expressibility\"]:.6f}')
        print(f'Min expressibility: {stats[\"min_expressibility\"]:.6f}')
"
```

## 🎛️ 第五步：自定义配置运行

### 5.1 创建自定义配置
```bash
cat > configs/my_research.yaml << EOF
project:
  name: "my_quantum_research"
  base_output_dir: "research_results"
  timestamp_dirs: true

circuit_generation:
  num_circuits: 500       # 更多电路
  num_qubits: 6           # 更多量子比特
  max_depth: 30
  max_gates: 40
  tfim_J: 1.0
  tfim_g: 0.5             # 不同的参数

vqe:
  enabled: true
  batch_size: 50
  n_repeat: 3             # 每个电路重复3次，选最好结果
  optimizer: "L_BFGS_B"
  verbose: false

expressibility:
  enabled: true
  samples: 5000           # 更多样本，更精确
  bins: 75

noisy_vqe:
  enabled: true           # 启用噪声仿真
  batch_size: 25          # 噪声仿真较慢，减少批大小
  single_qubit_error: 0.001
  two_qubit_error: 0.01

logging:
  level: "INFO"
  format_type: "human"

progress:
  show_overall_progress: true
  show_individual_progress: true
EOF
```

### 5.2 运行自定义配置
```bash
python scripts/run_simulation.py --config configs/my_research.yaml
```

### 5.3 使用命令行覆盖参数
```bash
# 只运行VQE，不计算表达性
python scripts/run_simulation.py --config configs/my_research.yaml --vqe-only

# 使用更多重复次数
python scripts/run_simulation.py --config configs/my_research.yaml --vqe-n-repeat 5

# 调整电路参数
python scripts/run_simulation.py --config configs/my_research.yaml \
    --num-circuits 200 --num-qubits 8 --vqe-batch-size 25
```

## 🔄 第六步：使用已有电路

### 6.1 使用之前生成的电路
```bash
# 列出已有的电路文件
ls output/*/data/circuits/

# 使用特定的电路文件运行新仿真
python scripts/run_simulation.py \
    --circuits output/quick_test_quantum_vqe_20241202_143022/data/circuits/circuits_20241202_143022.pkl \
    --expressibility-only --expr-samples 10000
```

### 6.2 只运行特定类型的仿真
```bash
# 只计算表达性（高精度）
python scripts/run_simulation.py \
    --circuits existing_circuits.pkl \
    --expressibility-only \
    --expr-samples 20000 \
    --expr-bins 100

# 只运行噪声VQE比较
python scripts/run_simulation.py \
    --circuits existing_circuits.pkl \
    --noisy-vqe-only \
    --output-dir noisy_comparison
```

## 📈 第七步：结果分析和可视化

### 7.1 使用内置分析脚本
```bash
# 分析VQE结果
python scripts/analyze_results.py \
    --input_dir output/my_quantum_research_20241202_150000 \
    --visualize \
    --export_csv

# 只生成可视化图表
python scripts/analyze_results.py \
    --input_dir output/my_quantum_research_20241202_150000 \
    --visualize \
    --output_dir analysis_plots
```

### 7.2 自定义分析脚本
```python
# custom_analysis.py
import json
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
with open('output/my_quantum_research_20241202_150000/data/results/pipeline_results_20241202_150500.json') as f:
    data = json.load(f)

# 分析VQE结果
vqe_results = data['simulation_results']['vqe']['results']
energies = [r['result_data']['energy'] for r in vqe_results if r['status'] == 'success']
exact_energy = data['circuit_info']['exact_ground_energy']

# 绘制能量分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(energies, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(exact_energy, color='red', linestyle='--', label=f'Exact: {exact_energy:.6f}')
plt.xlabel('VQE Energy')
plt.ylabel('Count')
plt.title('VQE Energy Distribution')
plt.legend()

# 绘制误差分布
errors = [abs(e - exact_energy) for e in energies]
plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('|VQE Energy - Exact Energy|')
plt.ylabel('Count')
plt.title('VQE Error Distribution')
plt.yscale('log')

plt.tight_layout()
plt.savefig('vqe_analysis.png', dpi=300)
plt.show()

print(f"Analysis complete:")
print(f"  Total successful VQE runs: {len(energies)}")
print(f"  Best energy found: {min(energies):.6f}")
print(f"  Mean error: {np.mean(errors):.6f}")
print(f"  Median error: {np.median(errors):.6f}")
```

## 🎯 第八步：生产环境配置

### 8.1 创建生产配置
```bash
python scripts/run_simulation.py --create-config production
```

### 8.2 编辑生产配置
```yaml
# configs/production.yaml
project:
  name: "production_quantum_vqe"
  base_output_dir: "production_results"

circuit_generation:
  num_circuits: 10000     # 大规模
  num_qubits: 8
  max_depth: 50
  max_gates: 60

vqe:
  enabled: true
  batch_size: 200         # 大批次
  max_workers: 16         # 多进程
  n_repeat: 5             # 高重复次数

expressibility:
  enabled: true
  samples: 10000          # 高精度
  bins: 100

noisy_vqe:
  enabled: true
  batch_size: 100

logging:
  level: "WARNING"        # 减少日志
  file_rotation: true

storage:
  compression: true       # 启用压缩节省空间
  backup_enabled: true
```

### 8.3 运行生产任务
```bash
# 后台运行大规模任务
nohup python scripts/run_simulation.py --config configs/production.yaml > production.log 2>&1 &

# 监控进度
tail -f production.log
tail -f production_results/*/logs/*.log
```

## 🔍 故障排除示例

### 常见问题和解决方案

```bash
# 1. 内存不足
python scripts/run_simulation.py --config configs/production.yaml \
    --vqe-batch-size 50 --vqe-max-workers 8

# 2. 磁盘空间不足
python scripts/run_simulation.py --config configs/production.yaml \
    --num-circuits 5000

# 3. 只测试VQE性能
python scripts/run_simulation.py --config configs/quick_test.yaml \
    --vqe-only --vqe-verbose

# 4. 调试配置问题
python scripts/run_simulation.py --validate-config --config your_config.yaml
python scripts/run_simulation.py --dry-run --config your_config.yaml
```

## 🎉 完成！

您现在已经完整体验了新的量子VQE仿真流程！

**主要收获**：
- ✅ 零代码修改的配置驱动工作流程
- ✅ 实时进度跟踪和详细日志
- ✅ 自动结果管理和分析
- ✅ 灵活的参数覆盖和选择性执行
- ✅ 生产就绪的并行处理和错误处理

**下一步探索**：
- 尝试不同的TFIM参数组合
- 实验不同的优化器和噪声模型
- 开发自定义分析脚本
- 集成到您的研究工作流程中

有任何问题或需要进一步定制，随时告诉我！🚀
