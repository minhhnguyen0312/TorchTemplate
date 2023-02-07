import torch

from torch.utils.data import Dataset, DataLoader
from .transforms import Transform as T

class BaseDataset(Dataset):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self.src = config['source']
        self.transform = T(config['transform'])
        self.items = []
        self.retrieve_data()

    def retrieve_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.items)

    def getitem(self):
        pass

class BaseInput:
    def __init__(self):
        pass
