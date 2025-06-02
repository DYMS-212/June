# src/utils/logger.py
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from logging.handlers import RotatingFileHandler
import multiprocessing as mp


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器，输出JSON格式的日志"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加额外的上下文信息
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
            
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """人类可读的日志格式器"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class QuantumVQELogger:
    """量子VQE项目的统一日志管理器"""
    
    def __init__(self, 
                 name: str = "quantum_vqe",
                 log_dir: str = "logs",
                 level: str = "INFO",
                 format_type: str = "human",  # "human" or "structured"
                 file_rotation: bool = True,
                 max_size: str = "100MB",
                 backup_count: int = 5,
                 enable_console: bool = True):
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        self.format_type = format_type
        self.file_rotation = file_rotation
        self.max_size = self._parse_size(max_size)
        self.backup_count = backup_count
        self.enable_console = enable_console
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化根日志器
        self.logger = self._setup_logger()
        
        # 存储子日志器
        self._child_loggers = {}
        
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串，返回字节数"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def _setup_logger(self) -> logging.Logger:
        """设置主日志器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 选择格式器
        if self.format_type == "structured":
            formatter = StructuredFormatter()
            file_suffix = ".jsonl"
        else:
            formatter = HumanReadableFormatter()
            file_suffix = ".log"
        
        # 文件处理器
        log_file = self.log_dir / f"{self.name}{file_suffix}"
        if self.file_rotation:
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=self.max_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            # 控制台总是使用人类可读格式
            console_handler.setFormatter(HumanReadableFormatter())
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取子日志器"""
        full_name = f"{self.name}.{name}"
        if full_name not in self._child_loggers:
            child_logger = logging.getLogger(full_name)
            child_logger.setLevel(self.level)
            self._child_loggers[full_name] = child_logger
        
        return self._child_loggers[full_name]
    
    def log_with_context(self, level: str, message: str, **context):
        """带上下文的日志记录"""
        extra_data = {'extra_data': context} if context else {}
        getattr(self.logger, level.lower())(message, extra=extra_data)
    
    def log_simulation_start(self, sim_type: str, config: Dict[str, Any]):
        """记录仿真开始"""
        self.log_with_context(
            'info',
            f"Starting {sim_type} simulation",
            simulation_type=sim_type,
            config=config,
            timestamp=datetime.now().isoformat()
        )
    
    def log_simulation_progress(self, sim_type: str, progress: float, details: Dict[str, Any] = None):
        """记录仿真进度"""
        context = {
            'simulation_type': sim_type,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        }
        if details:
            context.update(details)
            
        self.log_with_context(
            'info',
            f"{sim_type} simulation progress: {progress:.1%}",
            **context
        )
    
    def log_simulation_complete(self, sim_type: str, results: Dict[str, Any]):
        """记录仿真完成"""
        self.log_with_context(
            'info',
            f"Completed {sim_type} simulation",
            simulation_type=sim_type,
            results=results,
            timestamp=datetime.now().isoformat()
        )
    
    def log_error(self, message: str, error: Exception = None, **context):
        """记录错误"""
        if error:
            context['error_type'] = type(error).__name__
            context['error_message'] = str(error)
        
        self.log_with_context('error', message, **context)
        
        if error:
            self.logger.exception(f"Exception in {message}")
    
    def close(self):
        """关闭日志器"""
        for handler in self.logger.handlers:
            handler.close()
        
        for child_logger in self._child_loggers.values():
            for handler in child_logger.handlers:
                handler.close()


# 全局日志管理器实例
_global_logger: Optional[QuantumVQELogger] = None


def setup_global_logger(config: Dict[str, Any]) -> QuantumVQELogger:
    """设置全局日志管理器"""
    global _global_logger
    _global_logger = QuantumVQELogger(**config)
    return _global_logger


def get_logger(name: str = None) -> logging.Logger:
    """获取日志器实例"""
    if _global_logger is None:
        # 如果全局日志器未初始化，创建默认的
        setup_global_logger({})
    
    if name:
        return _global_logger.get_logger(name)
    else:
        return _global_logger.logger


def log_with_context(level: str, message: str, **context):
    """便捷的上下文日志记录函数"""
    if _global_logger:
        _global_logger.log_with_context(level, message, **context)


# 便捷函数
def log_simulation_start(sim_type: str, config: Dict[str, Any]):
    if _global_logger:
        _global_logger.log_simulation_start(sim_type, config)


def log_simulation_progress(sim_type: str, progress: float, details: Dict[str, Any] = None):
    if _global_logger:
        _global_logger.log_simulation_progress(sim_type, progress, details)


def log_simulation_complete(sim_type: str, results: Dict[str, Any]):
    if _global_logger:
        _global_logger.log_simulation_complete(sim_type, results)


def log_error(message: str, error: Exception = None, **context):
    if _global_logger:
        _global_logger.log_error(message, error, **context)


# 多进程安全的日志记录器
class MultiProcessLogger:
    """多进程安全的日志记录器"""
    
    def __init__(self, base_logger: QuantumVQELogger):
        self.base_logger = base_logger
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()
        
    def get_process_logger(self, name: str = None) -> logging.Logger:
        """获取当前进程的日志器"""
        process_name = f"process_{self.process_id}"
        if name:
            process_name += f".{name}"
        
        return self.base_logger.get_logger(process_name)
    
    def log_from_process(self, level: str, message: str, **context):
        """从当前进程记录日志"""
        logger = self.get_process_logger()
        context['process_id'] = self.process_id
        context['thread_id'] = self.thread_id
        
        extra_data = {'extra_data': context} if context else {}
        getattr(logger, level.lower())(message, extra=extra_data)


if __name__ == "__main__":
    # 测试代码
    logger_manager = QuantumVQELogger(
        name="test_quantum_vqe",
        format_type="structured",
        level="DEBUG"
    )
    
    # 测试基本日志记录
    logger = logger_manager.get_logger("test")
    logger.info("这是一条测试信息")
    logger.warning("这是一条警告")
    
    # 测试上下文日志
    logger_manager.log_with_context(
        'info',
        "测试上下文日志",
        circuit_count=1000,
        simulation_type="VQE"
    )
    
    # 测试仿真日志
    logger_manager.log_simulation_start("VQE", {"num_circuits": 1000, "qubits": 8})
    logger_manager.log_simulation_progress("VQE", 0.5, {"completed": 500, "remaining": 500})
    logger_manager.log_simulation_complete("VQE", {"success_rate": 0.95, "avg_time": 2.3})
    
    print("日志测试完成，请检查 logs/ 目录下的文件")
