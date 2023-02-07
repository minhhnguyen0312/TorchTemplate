import os
from PIL import Image
from .base import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, config):
        super(Dataset, self).__init__(config)
    
    def retrieve_data(self):
        cls = os.listdir(self.src)
        for i, c in enumerate(cls):
            self.items += [(f"{self.src}/{c}/{p}", i)
                            for p in os.listdir(f"{self.src}/{c}")]
            self.idx2cls[c] = i
    
    def getitem(self, index):
        path, cls = self.items[index]
        image = Image.open(path).convert("RGB")