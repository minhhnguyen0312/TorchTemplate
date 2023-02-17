import torch
import torch.nn as nn
from torchvision.models import resnet18
from ..base_model import BaseClassifierModel

class ResnetBinary(BaseClassifierModel):
    def __init__(self, config):
        super(ResnetBinary, self).__init__(config)
        self.model = resnet18()
        self.num_classes = config['num_classes']
        self.classifier = nn.Linear(1000, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.initialize()

    def forward(self, image):
        x = self.model(image)
        logits = self.classifier(x)
        return logits
    