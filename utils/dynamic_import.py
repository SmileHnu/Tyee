import importlib

def dynamic_import(module_name, class_name):
        try:
            # 动态导入模块
            module = importlib.import_module(module_name)
            # 获取类，如果不存在则抛出 AttributeError
            cls = getattr(module, class_name)
            return cls
        except ImportError as e:
            raise ImportError(f"无法导入模块 '{module_name}': {e}")
        except AttributeError:
            raise AttributeError(f"模块 '{module_name}' 中没有找到类 '{class_name}'")