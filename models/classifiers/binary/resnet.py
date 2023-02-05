import torch
import torch.nn as nn
from torchvision.models import resnet18
from ..base_model import BaseClassifierModel

class ResnetBinary(BaseClassifierModel):
    def __init__(self):
        super(ResnetBinary, self).__init__()
        self.model = resnet18()
        self.classifier = nn.Linear(1000, 1)
        super().initialize()

    def forward(self, image):
        x = self.model(image)
        y = self.classifier(x)
        return nn.Sigmoid()(y)
    
    # def forward(self, batch):
    #     x, y = batch["images"], batch["classes"]
    #     yhat = self.predict(x)
    #     loss = self.criterion(y, yhat)
    #     return loss