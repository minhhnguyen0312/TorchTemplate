import os
import importlib

def mkdir(filepath):
    if os.path.exists(filepath):
        return
    else:
        os.mkdir(filepath)

def import_module(module: str):
    try:
        importlib.import_module(module)
    except ImportError:
        raise ImportError(f"Could not import module {module}.")

