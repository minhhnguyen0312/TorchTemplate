import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, config):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x.view(-1, 1, 28, 28)
        return x
