# src/utils/storage.py
import os
import json
import pickle
import gzip
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from .logger import get_logger


@dataclass
class StorageMetadata:
    """存储元数据"""
    file_path: str
    format: str
    created_at: str
    size_bytes: int
    checksum: str
    version: str = "1.0"
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageMetadata':
        return cls(**data)


class StorageManager:
    """统一存储管理器"""
    
    def __init__(self, 
                 base_dir: str = "data",
                 compression: bool = True,
                 backup_enabled: bool = True,
                 metadata_enabled: bool = True):
        
        self.base_dir = Path(base_dir)
        self.compression = compression
        self.backup_enabled = backup_enabled
        self.metadata_enabled = metadata_enabled
        
        # 创建目录结构
        self.circuits_dir = self.base_dir / "circuits"
        self.results_dir = self.base_dir / "results"
        self.metadata_dir = self.base_dir / "metadata"
        self.backup_dir = self.base_dir / "backups"
        self.temp_dir = self.base_dir / "temp"
        
        for dir_path in [self.circuits_dir, self.results_dir, 
                        self.metadata_dir, self.backup_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("storage")
        
        # 元数据缓存
        self.metadata_cache: Dict[str, StorageMetadata] = {}
        self._load_metadata_cache()
    
    def _load_metadata_cache(self):
        """加载元数据缓存"""
        if not self.metadata_enabled:
            return
            
        metadata_index = self.metadata_dir / "index.json"
        if metadata_index.exists():
            try:
                with open(metadata_index, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata_cache = {
                        k: StorageMetadata.from_dict(v) 
                        for k, v in data.items()
                    }
                self.logger.info(f"Loaded {len(self.metadata_cache)} metadata entries")
            except Exception as e:
                self.logger.error(f"Failed to load metadata cache: {e}")
    
    def _save_metadata_cache(self):
        """保存元数据缓存"""
        if not self.metadata_enabled:
            return
            
        metadata_index = self.metadata_dir / "index.json"
        try:
            data = {k: v.to_dict() for k, v in self.metadata_cache.items()}
            with open(metadata_index, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save metadata cache: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """创建文件备份"""
        if not self.backup_enabled or not file_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def _save_metadata(self, file_path: Path, format_type: str, description: str = ""):
        """保存文件元数据"""
        if not self.metadata_enabled or not file_path.exists():
            return
        
        file_size = file_path.stat().st_size
        checksum = self._calculate_checksum(file_path)
        
        metadata = StorageMetadata(
            file_path=str(file_path.relative_to(self.base_dir)),
            format=format_type,
            created_at=datetime.now().isoformat(),
            size_bytes=file_size,
            checksum=checksum,
            description=description
        )
        
        self.metadata_cache[str(file_path)] = metadata
        self._save_metadata_cache()
    
    def save_circuits(self, 
                     circuits: List[Any], 
                     filename: str,
                     description: str = "",
                     compress: bool = None) -> Path:
        """
        保存量子电路列表
        
        Args:
            circuits: 量子电路列表
            filename: 文件名（不含扩展名）
            description: 描述信息
            compress: 是否压缩，None则使用默认设置
        
        Returns:
            保存的文件路径
        """
        if compress is None:
            compress = self.compression
        
        # 确定文件路径和格式
        if compress:
            file_path = self.circuits_dir / f"{filename}.pkl.gz"
            format_type = "pkl.gz"
        else:
            file_path = self.circuits_dir / f"{filename}.pkl"
            format_type = "pkl"
        
        # 创建备份
        self._create_backup(file_path)
        
        # 保存电路
        try:
            if compress:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(circuits, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(circuits, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Saved {len(circuits)} circuits to {file_path}")
            
            # 保存元数据
            desc = description or f"Quantum circuits collection ({len(circuits)} circuits)"
            self._save_metadata(file_path, format_type, desc)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save circuits: {e}")
            raise
    
    def load_circuits(self, filename: str) -> List[Any]:
        """
        加载量子电路列表
        
        Args:
            filename: 文件名（可含或不含扩展名）
        
        Returns:
            量子电路列表
        """
        # 尝试不同的文件扩展名
        possible_paths = [
            self.circuits_dir / filename,
            self.circuits_dir / f"{filename}.pkl",
            self.circuits_dir / f"{filename}.pkl.gz",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(f"Circuit file not found: {filename}")
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    circuits = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    circuits = pickle.load(f)
            
            self.logger.info(f"Loaded {len(circuits)} circuits from {file_path}")
            return circuits
            
        except Exception as e:
            self.logger.error(f"Failed to load circuits: {e}")
            raise
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    filename: str,
                    format_type: str = "json",
                    description: str = "") -> Path:
        """
        保存仿真结果
        
        Args:
            results: 结果数据
            filename: 文件名（不含扩展名）
            format_type: 格式类型 ("json", "pkl", "hdf5")
            description: 描述信息
        
        Returns:
            保存的文件路径
        """
        if format_type == "json":
            return self._save_json_results(results, filename, description)
        elif format_type == "pkl":
            return self._save_pkl_results(results, filename, description)
        elif format_type == "hdf5":
            return self._save_hdf5_results(results, filename, description)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _save_json_results(self, results: Dict[str, Any], filename: str, description: str) -> Path:
        """保存JSON格式结果"""
        file_path = self.results_dir / f"{filename}.json"
        
        # 创建备份
        self._create_backup(file_path)
        
        try:
            # 处理特殊类型的数据
            serializable_results = self._make_json_serializable(results)
            
            if self.compression:
                with gzip.open(str(file_path) + '.gz', 'wt', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                file_path = Path(str(file_path) + '.gz')
                format_type = "json.gz"
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                format_type = "json"
            
            self.logger.info(f"Saved JSON results to {file_path}")
            self._save_metadata(file_path, format_type, description)
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            raise
    
    def _save_pkl_results(self, results: Dict[str, Any], filename: str, description: str) -> Path:
        """保存PKL格式结果"""
        file_path = self.results_dir / f"{filename}.pkl"
        
        # 创建备份
        self._create_backup(file_path)
        
        try:
            if self.compression:
                with gzip.open(str(file_path) + '.gz', 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                file_path = Path(str(file_path) + '.gz')
                format_type = "pkl.gz"
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                format_type = "pkl"
            
            self.logger.info(f"Saved PKL results to {file_path}")
            self._save_metadata(file_path, format_type, description)
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save PKL results: {e}")
            raise
    
    def _save_hdf5_results(self, results: Dict[str, Any], filename: str, description: str) -> Path:
        """保存HDF5格式结果"""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")
        
        file_path = self.results_dir / f"{filename}.h5"
        
        # 创建备份
        self._create_backup(file_path)
        
        try:
            with h5py.File(file_path, 'w') as f:
                self._write_dict_to_hdf5(f, results)
            
            self.logger.info(f"Saved HDF5 results to {file_path}")
            self._save_metadata(file_path, "hdf5", description)
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save HDF5 results: {e}")
            raise
    
    def _write_dict_to_hdf5(self, group, data_dict: Dict[str, Any]):
        """递归写入字典到HDF5"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_hdf5(subgroup, value)
            elif isinstance(value, (list, np.ndarray)):
                try:
                    group.create_dataset(key, data=value)
                except TypeError:
                    # 如果无法直接存储，转换为字符串
                    group.attrs[key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            else:
                # 其他类型转换为字符串
                group.attrs[key] = str(value)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        加载仿真结果
        
        Args:
            filename: 文件名（可含或不含扩展名）
        
        Returns:
            结果数据
        """
        # 尝试不同的文件格式
        possible_paths = [
            (self.results_dir / filename, None),
            (self.results_dir / f"{filename}.json", "json"),
            (self.results_dir / f"{filename}.json.gz", "json.gz"),
            (self.results_dir / f"{filename}.pkl", "pkl"),
            (self.results_dir / f"{filename}.pkl.gz", "pkl.gz"),
            (self.results_dir / f"{filename}.h5", "hdf5"),
        ]
        
        file_path = None
        format_type = None
        for path, fmt in possible_paths:
            if path.exists():
                file_path = path
                format_type = fmt or self._detect_format(path)
                break
        
        if file_path is None:
            raise FileNotFoundError(f"Results file not found: {filename}")
        
        try:
            if format_type.startswith("json"):
                return self._load_json_results(file_path, format_type)
            elif format_type.startswith("pkl"):
                return self._load_pkl_results(file_path, format_type)
            elif format_type == "hdf5":
                return self._load_hdf5_results(file_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise
    
    def _load_json_results(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """加载JSON格式结果"""
        try:
            if format_type == "json.gz":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            
            self.logger.info(f"Loaded JSON results from {file_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON results: {e}")
            raise
    
    def _load_pkl_results(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """加载PKL格式结果"""
        try:
            if format_type == "pkl.gz":
                with gzip.open(file_path, 'rb') as f:
                    results = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
            
            self.logger.info(f"Loaded PKL results from {file_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load PKL results: {e}")
            raise
    
    def _load_hdf5_results(self, file_path: Path) -> Dict[str, Any]:
        """加载HDF5格式结果"""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")
        
        try:
            with h5py.File(file_path, 'r') as f:
                results = self._read_hdf5_to_dict(f)
            
            self.logger.info(f"Loaded HDF5 results from {file_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load HDF5 results: {e}")
            raise
    
    def _read_hdf5_to_dict(self, group) -> Dict[str, Any]:
        """递归读取HDF5到字典"""
        result = {}
        
        # 读取属性
        for key, value in group.attrs.items():
            result[key] = value
        
        # 读取数据集和子组
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._read_hdf5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = item[...]
        
        return result
    
    def _detect_format(self, file_path: Path) -> str:
        """检测文件格式"""
        suffixes = file_path.suffixes
        if '.gz' in suffixes:
            if '.json' in suffixes:
                return "json.gz"
            elif '.pkl' in suffixes:
                return "pkl.gz"
        elif file_path.suffix == '.json':
            return "json"
        elif file_path.suffix == '.pkl':
            return "pkl"
        elif file_path.suffix == '.h5':
            return "hdf5"
        else:
            return "unknown"
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return self._make_json_serializable(obj.to_dict())
        else:
            return obj
    
    def list_files(self, directory: str = "all") -> List[Dict[str, Any]]:
        """列出存储的文件"""
        if directory == "all":
            dirs = [self.circuits_dir, self.results_dir]
        elif directory == "circuits":
            dirs = [self.circuits_dir]
        elif directory == "results":
            dirs = [self.results_dir]
        else:
            raise ValueError("Directory must be 'all', 'circuits', or 'results'")
        
        files = []
        for dir_path in dirs:
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    metadata = self.metadata_cache.get(str(file_path))
                    file_info = {
                        'path': str(file_path.relative_to(self.base_dir)),
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'type': directory if directory != "all" else file_path.parent.name
                    }
                    if metadata:
                        file_info.update(metadata.to_dict())
                    files.append(file_info)
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        for temp_file in self.temp_dir.iterdir():
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            except Exception as e:
                self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            'total_files': len(self.metadata_cache),
            'circuits_count': 0,
            'results_count': 0,
            'total_size': 0,
            'formats': {}
        }
        
        for metadata in self.metadata_cache.values():
            stats['total_size'] += metadata.size_bytes
            
            if 'circuits' in metadata.file_path:
                stats['circuits_count'] += 1
            elif 'results' in metadata.file_path:
                stats['results_count'] += 1
            
            format_key = metadata.format
            if format_key not in stats['formats']:
                stats['formats'][format_key] = {'count': 0, 'size': 0}
            stats['formats'][format_key]['count'] += 1
            stats['formats'][format_key]['size'] += metadata.size_bytes
        
        return stats


if __name__ == "__main__":
    # 测试代码
    from .logger import setup_global_logger
    
    # 设置日志
    setup_global_logger({
        'name': 'test_storage',
        'format_type': 'human',
        'level': 'INFO'
    })
    
    # 创建存储管理器
    storage = StorageManager(base_dir="test_data")
    
    # 测试保存和加载
    test_data = {
        'simulation_type': 'VQE',
        'results': [
            {'energy': -1.23, 'time': 0.5},
            {'energy': -1.45, 'time': 0.7}
        ],
        'metadata': {
            'qubits': 8,
            'circuits': 100
        }
    }
    
    # 保存为不同格式
    storage.save_results(test_data, "test_vqe_results", "json", "Test VQE simulation results")
    storage.save_results(test_data, "test_vqe_results_pkl", "pkl", "Test VQE simulation results (pickle)")
    
    # 加载测试
    loaded_json = storage.load_results("test_vqe_results")
    loaded_pkl = storage.load_results("test_vqe_results_pkl")
    
    print("JSON数据:", loaded_json)
    print("PKL数据:", loaded_pkl)
    
    # 列出文件
    print("\n存储的文件:")
    files = storage.list_files()
    for file_info in files:
        print(f"- {file_info['name']} ({file_info['size']} bytes)")
    
    # 存储统计
    stats = storage.get_storage_stats()
    print(f"\n存储统计: {stats}")
    
    print("存储系统测试完成")
