from torch.utils.tensorboard import SummaryWriter
import time
class Writer(SummaryWriter):
    """Tensorboard Writer class
    Initialize with given metrics configuration.
    Params:
        - train_metrics: {
            'loss': self.add_scalar,
        }
    """
    def __init__(self, metric_cfg):
        super(Writer, self).__init__()
        self.log_step = 100
        self.cur_step = 0
        self.register(metric_cfg)
    
    def register(self, cfg):
        self.train_metrics = {}
        self.eval_metrics = {}
        for k, v in cfg['train'].items():
            self.train_metrics[k] = self.get_bounded_func(v)
        for k, v in cfg['eval'].items():
            self.train_metrics[k] = self.get_bounded_func(v)
    
    def get_bounded_func(self, dtype:str):
        if dtype in ["int", "float", "long", "double"]:
            return self.add_scalar
        else:
            raise ValueError(f"Writer are not implemented for dtype {dtype}")
    
    
    def update(self, outputs, isval=False):
        if self.cur_step % self.log_step == 0:
            if not isval:
                self.cur_step += 1
                ref = self.train_metrics
            else:
                ref = self.eval_metrics
            for k, v in outputs.items():
                ref[k](k, v, self.cur_step)

class Timer:
    def __init__(self, msg=""):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(self.msg, f"Runtime: {self.end_time - self.start:.2f} seconds.")