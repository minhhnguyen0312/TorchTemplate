from tqdm import tqdm

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config
    
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
            casted_inputs[k] = v.to(self.device)

    def train_step(self, batch, step):
        raise NotImplementedError
    
    def eval_step(self, batch, step):
        raise NotImplementedError

    def train_one_epoch(self, data, epoch):
        self.train()
        with tqdm(data) as t:
            t.set_description(f"Training {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.train_step(batch, i)
                if self.config['log']:
                    self.writer.update(outputs, isval=False)
    
    def eval_one_epoch(self, data, epoch):
        self.eval()
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

    