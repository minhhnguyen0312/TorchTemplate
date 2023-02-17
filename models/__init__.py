import importlib

def build_model_from_config(config):
    model_module = ".".join(config['module'].split('.')[:-1])
    cls = config['module'].split('.')[-1]
    mod = getattr(importlib.import_module(model_module), cls)
    return mod(config)
