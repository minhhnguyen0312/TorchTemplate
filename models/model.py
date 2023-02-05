from tqdm import tqdm

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
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
                self.writer.update(outputs, isval=False)
    
    def eval_one_epoch(self, data, epoch):
        self.eval()
        with tqdm(data) as t:
            t.set_description(f"Validating {epoch}:")
            for i, batch in enumerate(t):
                outputs = self.eval_step(batch, i)
                self.writer.update(outputs, ival=True)
    