import torch
import torch.nn as nn
from ..model import BaseModel

class BaseClassifierModel(BaseModel):
    def __init__(self):
        super(BaseClassifierModel, self).__init__()

    def initialize(self):
        self.optimizer = torch.optim.Adam(self.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
    
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

