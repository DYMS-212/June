#!/usr/bin/env python3
# quick_fix.py - 快速修复脚本

"""
快速修复VQE仿真器中的属性初始化问题
"""

def fix_vqe_simulator():
    """修复VQE仿真器文件"""
    
    vqe_file = "src/simulators/vqe_simulator.py"
    
    try:
        with open(vqe_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有original_loggers初始化
        if "self.original_loggers = {}" not in content:
            # 在__init__方法中添加初始化
            old_line = "self.original_loggers = {}"
            new_lines = """        # 用于抑制输出的设置
        self.null_io = NullIO()
        self.original_loggers = {}  # 初始化为空字典
        self.logging_filters = []   # 存储添加的过滤器
        
        # 标记是否已设置日志抑制
        self.logging_suppression_active = False"""
            
            # 查找插入位置
            init_pattern = "# 用于抑制输出的设置"
            if init_pattern in content:
                content = content.replace(
                    "# 用于抑制输出的设置\n        self.null_io = NullIO()\n        self.original_loggers = {}",
                    new_lines
                )
            else:
                # 如果找不到特定模式，在__init__末尾添加
                init_end = "super().__init__(config, storage_manager, hamiltonian, **kwargs)"
                content = content.replace(
                    init_end,
                    init_end + "\n        \n" + new_lines
                )
        
        # 确保cleanup方法是安全的
        cleanup_fix = """    def cleanup(self):
        \"\"\"清理资源\"\"\"
        try:
            # 恢复日志设置
            if hasattr(self, '_restore_logging'):
                self._restore_logging()
            
            # 清理其他资源
            if hasattr(self, 'estimator'):
                self.estimator = None
            if hasattr(self, 'optimizer'):
                self.optimizer = None
            
            self.logger.info("VQE simulator cleanup complete")
            
        except Exception as e:
            # 即使清理失败也要继续，但记录错误
            if hasattr(self, 'logger'):
                self.logger.warning(f"VQE simulator cleanup had issues: {e}")
            else:
                print(f"VQE simulator cleanup had issues: {e}")"""
        
        # 查找并替换cleanup方法
        import re
        cleanup_pattern = r'def cleanup\(self\):.*?(?=\n    def|\nclass|\n# |\Z)'
        if re.search(cleanup_pattern, content, re.DOTALL):
            content = re.sub(cleanup_pattern, cleanup_fix.strip(), content, flags=re.DOTALL)
        else:
            # 如果没找到cleanup方法，在类的末尾添加
            content += "\n" + cleanup_fix
        
        # 写入修复后的文件
        with open(vqe_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 已修复 {vqe_file}")
        return True
        
    except FileNotFoundError:
        print(f"✗ 文件未找到: {vqe_file}")
        print("请确保您在正确的项目目录中运行此脚本")
        return False
    except Exception as e:
        print(f"✗ 修复失败: {e}")
        return False


def create_minimal_fix():
    """创建最小修复补丁"""
    
    patch_content = """
# 临时修复 - 在您的VQE仿真器__init__方法中添加以下行：

# 在 super().__init__(...) 之后添加：
self.original_loggers = {}
self.logging_filters = []
self.logging_suppression_active = False

# 在 cleanup 方法中使用安全检查：
def cleanup(self):
    try:
        if hasattr(self, '_restore_logging'):
            self._restore_logging()
        if hasattr(self, 'estimator'):
            self.estimator = None
        if hasattr(self, 'optimizer'):
            self.optimizer = None
        self.logger.info("VQE simulator cleanup complete")
    except Exception as e:
        if hasattr(self, 'logger'):
            self.logger.warning(f"Cleanup issues: {e}")
"""
    
    with open("vqe_fix_patch.txt", "w") as f:
        f.write(patch_content)
    
    print("✓ 已创建修复补丁文件: vqe_fix_patch.txt")


def test_fix():
    """测试修复是否成功"""
    
    try:
        import sys
        sys.path.insert(0, "src")
        
        from simulators.vqe_simulator import VQESimulator, VQESimulationConfig
        from utils.storage import StorageManager
        
        # 创建测试配置
        config = VQESimulationConfig(name="test")
        storage = StorageManager()
        
        # 测试创建仿真器（不需要真正的哈密顿量）
        simulator = VQESimulator(config, storage, None)
        
        # 测试cleanup
        simulator.cleanup()
        
        print("✓ 修复测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 修复测试失败: {e}")
        return False


def main():
    """主修复流程"""
    
    print("VQE仿真器快速修复工具")
    print("=" * 30)
    
    print("\n1. 尝试自动修复...")
    if fix_vqe_simulator():
        print("\n2. 测试修复...")
        if test_fix():
            print("\n✅ 修复完成！您现在可以重新运行仿真了。")
        else:
            print("\n⚠️ 自动修复可能不完整，请检查文件内容。")
            create_minimal_fix()
    else:
        print("\n2. 创建手动修复补丁...")
        create_minimal_fix()
        print("\n📝 请手动应用修复补丁，或使用我提供的修复版本文件。")
    
    print(f"\n🚀 修复提示:")
    print("- 如果问题仍然存在，请用我提供的 'vqe_simulator_fixed' 文件替换原文件")
    print("- 或者在VQE仿真器的__init__方法中添加: self.original_loggers = {}")
    print("- 在cleanup方法中添加hasattr检查")


if __name__ == "__main__":
    main()
