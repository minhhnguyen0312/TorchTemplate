import os
from utils.args import PythonParser, YmlParser
from models import build_model_from_config

class BaseTrainer:
    def __init__(self):
        self.model = None
        self.trn_data = None
        self.val_data = None
        self.initialize()
    
    def initialize(self):
        raise NotImplementedError

    def fit(self, num_epochs=5):
        for i in range(num_epochs):
            self.model.train_one_epoch(self.trn_data, i)
            self.val_data.train_one_epoch(self.val_data, i)


class TrainerYml(BaseTrainer):
    def __init__(self, taskfile):
        self.taskfile = taskfile
        self.parser = YmlParser()
        super(TrainerYml, self).__init__()
    
    def initialize(self):
        self.config = self.parser.read(self.taskfile)
        self.initialize_model()
        self.initialize_data()
    
    def initialize_model(self):
        self.model = build_model_from_config(self.config['model'])
    
    def initialize_data(self):
        pass