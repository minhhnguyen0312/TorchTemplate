import os
from tqdm import tqdm

import torch
import torch.nn as nn
from utils.log import Writer

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        if not os.path.exists(config['save']):
            os.makedirs(config['save'])
        self.savedir = config['save']
        self.writer = Writer(logdir=config['save'] + "/logs", metric=config['metric'])
    
    def initialize(self):
        if self.config['optimizer']['module'] == 'sgd':
            optmod = torch.optim.SDG
        else:
            optmod = torch.optim.Adam
        opt_config = self.config['optimizer']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt_config['lr'])
        sch_config = opt_config['scheduler']
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=sch_config['patience'], factor=sch_config['factor'])
    
    def cast_inputs(self, inputs):
        casted_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                casted_inputs[k] = v.to(self.device)
            else:
                casted_inputs[k] = v
        return casted_inputs

    def train_step(self, batch, step):
        raise NotImplementedError
    
    def eval_step(self, batch, step):
        raise NotImplementedError

    def train_one_epoch(self, data, epoch, **kwargs):
        total = {}
        count = 0
        with tqdm(data) as t:
            t.set_description(f"Training {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.train_step(batch, i)
                for k, v in outputs.items():
                    total[k] = total.get(k, 0) + v
                count += 1
                t.set_postfix(**outputs)
                if self.config['log']:
                    self.writer.update(outputs, isval=False)
        for k, v in total.items():
            total[k] = v / count
        self.on_epoch_end(outputs=total, **kwargs)
    
    def eval_one_epoch(self, data, epoch):
        with tqdm(data) as t:
            t.set_description(f"Validating {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.eval_step(batch, i)
                if self.config['log']:
                    self.writer.update(outputs, ival=True)

    def save(self):
        filepath = f"{self.config['save']}/{self.config['taskname']}/latest.pt"
        torch.save(filepath, {
            'state_dict': self.state_dict(),
            'config': self.config
        })
    
    def on_epoch_end(self, **kwargs):
        # outputs = kwargs.pop(outputs)
        pass

    