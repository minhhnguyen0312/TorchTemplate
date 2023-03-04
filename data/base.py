import os
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
# from .transforms import Transform as T
from torchvision import transforms as T

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
    def __init__(self, config):
        self.dir = config['source']
        self.source = os.listdir(config['source'])
        self.train = True
        self.image_size = config['image_size']
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.source[index])
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        image_tensor = T.ToTensor()(image)
        
        return {
            "image": image_tensor,
        }

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        batch_ = {}
        for k in keys:
            batch_[k] = [item[k] for item in batch]
        return batch_