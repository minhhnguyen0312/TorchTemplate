import importlib

def build_model_from_config(config):
    model_module, cls = ".".join(config['module'].split('.')[:-1]), config['module'].split('.')[-1]
    mod = getattr(importlib.import_module(model_module), cls)
    opt_module, cls = ".".join(config['optimizer']['module'].split('.')[:-1]), config['optimizer']['module'].split('.')[-1]
    opt_mod = getattr(importlib.import_module(opt_module), cls)
    sch_module, cls = ".".join(config['optimizer']['scheduler']['module'].split('.')[:-1]), config['optimizer']['scheduler']['module'].split('.')[-1]
    sch_mod = getattr(importlib.import_module(sch_module), cls)

    return mod(config, opt_mod, sch_mod)
