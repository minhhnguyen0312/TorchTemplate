import os
from utils.args import PythonParser

class Trainer:
    def __init__(self):
        self.model = None
        self.trn_data = None
        self.val_data = None

    def fit(self, num_epochs=5):
        for i in range(num_epochs):
            self.model.train_one_epoch(self.trn_data, i)
            self.val_data.train_one_epoch(self.val_data, i)