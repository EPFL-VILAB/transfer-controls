import torch
import torchvision.transforms as transforms
import glob
import random
import pytorch_lightning as pl

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from data.taskonomy.taskonomy_datamodule import KEpochsSampler
from data.taskonomy import convertrgb2lab
from sklearn.model_selection import StratifiedShuffleSplit

class EurosatDataset(Dataset):
    """Eurosat encodable dataset class"""

    def __init__(self, 
                 data_dir, 
                 normalize,
                 max_images=1e6,
                 split='train', 
                 data_seed=0,
                 rgb2lab=False,
                 stratified=False,
                ):
        super().__init__()
        self.n_classes = 10
        
        path = f'{data_dir}/{split}/*/*.jpg'
        self.data = list(glob.glob(path))
        cats = list(set([path.split("/")[-2] for path in self.data]))
        cats.sort()
        self.labels = torch.LongTensor([cats.index(path.split("/")[-2]) for path in self.data])
        self.split = split
        if stratified:
            if split != 'test' and max_images < len(self.labels) - self.n_classes:
                ssplit = StratifiedShuffleSplit(n_splits=1, train_size=max_images, random_state=data_seed)
                indices, _ = next(ssplit.split(self.data, self.labels))
                self.data = [self.data[index] for index in indices]
                self.labels = self.labels[indices]
            
        if rgb2lab:
            rgb_transform = lambda x: convertrgb2lab(x)
        else:
            rgb_transform = lambda x: x

        self.train_preprocessor = transforms.Compose([
            rgb_transform,
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_preprocessor = transforms.Compose([
            rgb_transform,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])    
        self.encoded_data = {}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.split == 'train':
            preprocessor = self.train_preprocessor
        elif self.split == 'test' or self.split == 'val':
            preprocessor = self.test_preprocessor
        
        if idx not in self.encoded_data:
            self.encoded_data[idx] = Image.open(self.data[idx]).convert('RGB'), self.labels[idx]
        image = preprocessor(self.encoded_data[idx][0])
        return {'rgb': image, 'classification': self.encoded_data[idx][1]}
    
    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return int(max(self.labels) + 1)
    

class EurosatDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir='/PATH_TO/eurosat',
                 batch_size=32, 
                 num_workers=16,
                 data_seed=0,
                 pin_memory=True,
                 rgb2lab=False,
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
        self.stratified = stratified
        self.rgb2lab = False
        
        if kwargs['ssl_name'] == 'color':
            self.normalize  = transforms.Normalize(0.5, 0.01)
            self.rgb2lab = True
        elif 'task' not in kwargs['ssl_name']:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = lambda x: x
         
    def setup(self, stage=None):
        print(f"Batch size = {self.batch_size}, data_seed = {self.data_seed}")
        if stage == 'fit' or stage is None:
            self.trainset = EurosatDataset(data_dir=self.data_dir, 
                                normalize=self.normalize, max_images=self.max_images_train,
                                split='train', data_seed=self.data_seed, rgb2lab=self.rgb2lab,
                                stratified=self.stratified,
                            )
            self.valset = EurosatDataset(data_dir=self.data_dir,
                                normalize=self.normalize, max_images=self.max_images_val,
                                split='val', data_seed=self.data_seed, rgb2lab=self.rgb2lab,
                                stratified=self.stratified,
                            )
            
            if not self.stratified:
                max_images_train = min(self.max_images_train, len(self.trainset))
                max_images_val = min(self.max_images_val, len(self.valset))
                
                self.trainset, _ = random_split(self.trainset,
                    [max_images_train, len(self.trainset) - max_images_train],
                    generator=torch.Generator().manual_seed(self.data_seed)
                )
                
                self.valset, _ = random_split(self.valset,
                    [max_images_val, len(self.valset) - max_images_val],
                    generator=torch.Generator().manual_seed(self.data_seed)
                )
            
        if stage == 'test' or stage is None:
            self.testset = EurosatDataset(data_dir=self.data_dir, normalize=self.normalize,
                                split='test', rgb2lab=self.rgb2lab
                            )
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