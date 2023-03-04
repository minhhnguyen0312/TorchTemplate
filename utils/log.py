from torch.utils.tensorboard import SummaryWriter
import os

class Writer(SummaryWriter):
    """Tensorboard Writer class
    Initialize with given metrics configuration.
    Params:
        - train_metrics: {
            'loss': self.add_scalar,
        }
    """
    def __init__(self, logdir="", metric:dict = {}):
        super(Writer, self).__init__(log_dir=logdir)
        self.log_step = 100
        self.cur_step = 0
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.register(metric)
    
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