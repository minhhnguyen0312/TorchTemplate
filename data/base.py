import os
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


class BaseInput:
    def __init__(self):
        pass

class FFHQ_GAN(Dataset):
    DIR = "/kaggle/input/ffhq-face-data-set/"
    def __init__(self, config):
        self.source = os.listdir(DIR)
        
    def __len__(self):
        return len(self.source)
    
    def __getitem(self, index):
        path = os.path.join(DIR, self.source[index])
        image = Image.open(path).convert("RGB")
        image_tensor = T.ToTensor()(image)
        return image_tensor