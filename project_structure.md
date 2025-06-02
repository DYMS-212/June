# 量子电路VQE项目重构方案

## 推荐的项目结构

```
quantum_vqe_project/
├── src/                          # 核心源码
│   ├── __init__.py
│   ├── models/                   # 物理模型
│   │   ├── __init__.py
│   │   └── tfim.py              # TFIM模型
│   ├── generators/               # 电路生成器
│   │   ├── __init__.py
│   │   └── layerwise.py         # 分层电路生成
│   ├── simulators/               # 仿真器
│   │   ├── __init__.py
│   │   ├── base_simulator.py    # 基础仿真器类
│   │   ├── vqe_simulator.py     # 无噪声VQE
│   │   ├── noisy_vqe_simulator.py # 含噪声VQE
│   │   └── expressibility.py    # 表达性计算
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py            # 统一日志管理
│   │   ├── storage.py           # 文件存储管理
│   │   └── progress.py          # 进度条管理
│   └── pipeline.py               # 主流水线控制器
├── configs/                      # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── vqe_config.yaml          # VQE专用配置
│   └── noise_config.yaml        # 噪声模型配置
├── scripts/                      # 执行脚本
│   ├── generate_circuits.py     # 生成电路
│   ├── run_simulation.py        # 运行仿真
│   └── analyze_results.py       # 分析结果
├── data/                         # 数据目录
│   ├── circuits/                # 电路文件
│   ├── results/                 # 仿真结果
│   └── analysis/                # 分析结果
├── logs/                         # 日志文件
├── tests/                        # 测试文件
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```

## 主要改进点

### 1. 统一配置管理
- 使用YAML配置文件替代命令行参数
- 支持配置继承和覆盖
- 环境变量支持

### 2. 流水线式处理
- 明确分离电路生成和仿真步骤
- 支持选择性运行不同类型的仿真
- 自动依赖检查和数据传递

### 3. 统一的仿真器接口
```python
class BaseSimulator:
    def setup(self, config): pass
    def run_batch(self, circuits, batch_id): pass
    def get_results(self): pass
    def cleanup(self): pass
```

### 4. 改进的存储策略
- **电路**: 继续使用PKL（Qiskit对象序列化）
- **结果**: 使用JSON + HDF5混合存储
  - 元数据和摘要用JSON（便于查看和解析）
  - 大量数值数据用HDF5（高效压缩）
- **配置**: YAML格式（人类可读）

### 5. 日志和进度管理
- 统一的日志格式和级别
- 实时进度条显示
- 结构化日志（JSON格式便于解析）
- 支持多进程日志聚合

## 配置文件示例

```yaml
# configs/default.yaml
project:
  name: "quantum_vqe_simulation"
  version: "1.0.0"

circuit_generation:
  num_circuits: 5000
  num_qubits: 8
  max_depth: 50
  max_gates: 48
  gate_stddev: 1.35
  gate_bias: 0.5

simulations:
  vqe:
    enabled: true
    n_repeat: 3
    batch_size: 100
    max_workers: null
  
  expressibility:
    enabled: true
    samples: 5000
    bins: 75
  
  noisy_vqe:
    enabled: false
    noise_model:
      single_qubit_error: 0.001
      two_qubit_error: 0.01
      t1_time: 50e-6
      t2_time: 20e-6

storage:
  base_dir: "./data"
  format:
    circuits: "pkl"
    results: "json+hdf5"
    metadata: "json"

logging:
  level: "INFO"
  format: "structured"
  file_rotation: true
  max_size: "100MB"
```

## 核心流水线控制器

```python
class QuantumVQEPipeline:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logger()
        self.storage = StorageManager(self.config)
        
    def run_full_pipeline(self):
        """运行完整流水线"""
        # 1. 生成电路
        circuits = self.generate_circuits()
        
        # 2. 运行选定的仿真
        results = {}
        if self.config.simulations.vqe.enabled:
            results['vqe'] = self.run_vqe_simulation(circuits)
            
        if self.config.simulations.expressibility.enabled:
            results['expressibility'] = self.run_expressibility(circuits)
            
        if self.config.simulations.noisy_vqe.enabled:
            results['noisy_vqe'] = self.run_noisy_vqe(circuits)
        
        # 3. 保存和分析结果
        self.save_results(results)
        return results
    
    def run_selective_simulation(self, circuit_path, sim_types):
        """基于已有电路运行选择性仿真"""
        circuits = self.storage.load_circuits(circuit_path)
        # ...
```

这个重构方案的优势：
1. **模块化**: 每个组件职责清晰
2. **可扩展**: 容易添加新的仿真器或模型
3. **可配置**: 通过配置文件灵活控制行为
4. **可维护**: 统一的接口和错误处理
5. **可测试**: 每个模块都可以独立测试

你觉得这个整体架构如何？如果同意这个方向，我可以继续提供具体的实现代码。下一步我们可以专注于：
1. 统一的日志和进度条系统
2. 改进的存储格式
3. 具体的流水线实现

你想先从哪个部分开始？
