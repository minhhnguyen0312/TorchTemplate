import torch
import torch.nn as nn
from ..model import BaseModel

class BaseClassifierModel(BaseModel):
    def __init__(self, config):
        super(BaseClassifierModel, self).__init__(config)
    
    # def initialize(self):
    #     super().initialize()

    def train_step(self, batch, i):
        batch = self.cast_inputs(batch)
        pred = self(batch['images'])
        loss = self.criterion(batch['labels'], pred)
        loss.backward()
        return loss

    def eval_step(self, batch):
        batch = self.cast_inputs(batch)
        pred = self(batch['images'])
        pred = pred.argmax(dim=1)
        return self.evaluate(pred, batch['labels'])

