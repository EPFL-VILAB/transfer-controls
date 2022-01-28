import json
import glob
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from data.taskonomy.taskonomy_datamodule import KEpochsSampler

class CLEVRDataset(Dataset):
    """CLEVR Num Objects encodable dataset class"""

    def __init__(self, data_dir='/PATH_TO/CLEVR_v1.0', train=True):
        super().__init__()
        path = f'{data_dir}/images/train/*.png' if train else f'{data_dir}/images/val/*.png'
        self.data = glob.glob(path)
        self.data.sort()
        labels_path = f'{data_dir}/scenes/CLEVR_train_scenes.json' if train else \
                        f'{data_dir}/scenes/CLEVR_val_scenes.json'
        with open(labels_path) as f:
            scene_data = json.load(f)
        self.labels = torch.LongTensor([len(s['objects']) for s in scene_data['scenes']])
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.encoded_data = {}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx not in self.encoded_data:
            self.encoded_data[idx] = self.preprocessor(Image.open(self.data[idx]).convert('RGB')), self.labels[idx]
        return {'rgb': self.encoded_data[idx][0], 'classification': self.encoded_data[idx][1]}

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)

class CLEVRDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir='/PATH_TO/CLEVR_v1.0',
                 batch_size=32, 
                 num_workers=16,
                 data_seed=-1,
                 pin_memory=True,
                 max_images_train: int = 1e6,
                 max_images_val: int = 1e6,
                 max_images_test: int = 1e6,
                 n_passes_epoch: int = 0,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_images_train = max_images_train
        self.max_images_val = max_images_val
        self.data_seed = data_seed
        self.n_passes_epoch = n_passes_epoch
         
    def setup(self, stage=None):
        print(f"Batch size = {self.batch_size}, data_seed = {self.data_seed}")
        if stage == 'fit' or stage is None:
            full_set = CLEVRDataset(data_dir=self.data_dir, train=True)
            size = int(len(full_set) * 0.1)
            self.trainset, self.valset = random_split(full_set,
                [len(full_set) - size, size],
                generator=torch.Generator().manual_seed(self.data_seed)
            )
            max_images_train = min(self.max_images_train, len(self.trainset))
            max_images_val = min(self.max_images_val, len(self.valset))
            
            self.trainset = Subset(self.trainset, range(max_images_train))
            self.valset = Subset(self.valset, range(max_images_val))
            
        if stage == 'test' or stage is None:
            self.testset  = CLEVRDataset(data_dir=self.data_dir, train=False)
            
        print(f"Dataset Size = {len(self.trainset)}")
        
    def train_dataloader(self):
        sampler = KEpochsSampler(
            torch.utils.data.RandomSampler(self.trainset), self.n_passes_epoch,
        )
        ''' PyTorch Lightning method: Creates train set dataloader. '''
        loader = DataLoader(
            self.trainset, batch_size=self.batch_size, sampler=sampler,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        ''' PyTorch Lightning method: Creates validation set dataloader. '''
        loader = DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        ''' PyTorch Lightning method: Creates test set dataloader. '''
        loader = DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        return loader