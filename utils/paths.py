import os
import importlib

def mkdir(filepath):
    if os.path.exists(filepath):
        return
    else:
        os.mkdir(filepath)

def import_module(module: str):
    try:
        model_module = ".".join(module.split('.')[:-1])
        cls = module.split('.')[-1]
        mod = getattr(importlib.import_module(model_module), cls)
        return mod
    except ImportError:
        raise ImportError(f"Could not import module {module}.")

