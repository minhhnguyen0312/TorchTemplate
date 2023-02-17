import torch
import torch.nn as nn
from torchvision.models import resnet18
from ..base_model import BaseClassifierModel

class ResnetBinary(BaseClassifierModel):
    def __init__(self, config):
        super(ResnetBinary, self).__init__(config)
        self.model = resnet18()
        self.classifier = nn.Linear(1000, 1)
        self.criterion = nn.BCELoss()
        self.initialize()

    def forward(self, image):
        x = self.model(image)
        y = self.classifier(x)
        return nn.Sigmoid()(y)
    