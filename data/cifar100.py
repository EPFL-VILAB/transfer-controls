import os
import glob
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl
from data.taskonomy.taskonomy_datamodule import KEpochsSampler
from data.taskonomy import convertrgb2lab
from sklearn.model_selection import StratifiedShuffleSplit

class CIFAR100Dataset(Dataset):
    """CIFAR-100 encodable dataset class"""

    def __init__(self, 
                 data_dir, 
                 normalize, 
                 split='train', 
                 rgb2lab=False
                ):
        
        super().__init__()
        self.split = split
        path = f'{data_dir}/cifar-100/test' if split=='test' else f'{data_dir}/cifar-100/train'
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        self.data = dict[list(dict.keys())[4]]
        self.labels = dict[list(dict.keys())[2]]
        
        
        if rgb2lab:
            rgb_transform = lambda x: convertrgb2lab(x)
        else:
            rgb_transform = lambda x: x

        self.train_preprocessor = transforms.Compose([    
            transforms.ToPILImage(),
            rgb_transform,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            rgb_transform,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split == 'train':
            preprocessor = self.train_preprocessor
        elif self.split == 'test' or self.split == 'val':
            preprocessor = self.test_preprocessor
        
        x = self.data[idx].reshape((3, 32, 32))
        x = np.moveaxis(x, 0, 2)
        x = preprocessor(x)
        return {'rgb': x, 'classification': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir='/DATA_DIR',
                 batch_size=32, 
                 num_workers=8,
                 data_seed=0,
                 pin_memory=True,
                 max_images_train: int = 1e6,
                 max_images_val: int = 1e6,
                 max_images_test: int = 1e6,
                 n_passes_epoch: int = 0,
                 stratified=False,
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
        self.stratified=stratified
        self.rgb2lab = False
        
        if kwargs['ssl_name'] == 'color':
            self.normalize  = transforms.Normalize(0.5, 0.01)
            self.rgb2lab = True
            
        elif 'task' not in kwargs['ssl_name']:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = lambda x: x
         
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_set = CIFAR100Dataset(data_dir=self.data_dir, normalize=self.normalize,
                            split='train', rgb2lab=self.rgb2lab
                        )
            val_set = CIFAR100Dataset(data_dir=self.data_dir, normalize=self.normalize,
                            split='val', rgb2lab=self.rgb2lab
                        )
            
            if self.stratified:
                rate = len(full_set) / (self.max_images_train + self.max_images_val)
        
                max_images_train = int(rate * self.max_images_train) if rate < 1 else self.max_images_train
                max_images_val = len(full_set) - int(rate * self.max_images_train) if rate < 1 else self.max_images_val

                ssplit = StratifiedShuffleSplit(n_splits=1, train_size=max_images_train,
                                                test_size=max_images_val, random_state=self.data_seed)

                train_indices, val_indices = next(ssplit.split(full_set.data, full_set.labels))

                self.trainset = Subset(full_set, train_indices)
                self.valset = Subset(val_set, val_indices)
            
            else:
                size = int(len(full_set) * 0.2)
                self.trainset, self.valset = random_split(full_set,
                    [len(full_set) - size, size],
                    generator=torch.Generator().manual_seed(self.data_seed)
                )
                max_images_train = min(self.max_images_train, len(self.trainset))
                max_images_val = min(self.max_images_val, len(self.valset))

                self.trainset = Subset(self.trainset, range(max_images_train))
                self.valset = Subset(self.valset, range(max_images_val))
                self.valset = Subset(val_set, self.valset.indices) 
            
        if stage == 'test' or stage is None:
            self.testset  = CIFAR100Dataset(data_dir=self.data_dir, normalize=self.normalize,
                                split='test', rgb2lab=self.rgb2lab
                            )

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