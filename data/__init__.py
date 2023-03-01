from re import T
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from utils.paths import import_module
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TmpMnist(Dataset):
    def __init__(self, dataset):
        super(TmpMnist, self).__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        image, cls = self.dataset.__getitem__(key)
        return {
            'image': image
        }

def collate_fn(batch):
    k = batch[0].keys()
    return {
        k: torch.Tensor([item[k] for item in batch])
    }

def build_data_from_config(config):
    if config['source'] == "mnist":
        train_dataset = datasets.MNIST(root='./source', train=True, transform=transform, download=True)
        train_loader = DataLoader(TmpMnist(train_dataset), batch_size=16, shuffle=True)
        return train_loader, None
    else:
        mod = import_module(config['module'])
        dataset = mod(config)
        loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
        return loader