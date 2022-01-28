import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import glob
from multiprocessing import Manager

from .taskonomy_dataset import TaskonomyDataset
from ..asynchronous_loader import AsynchronousLoader


class KEpochsSampler(object):
    def __init__(self, base_iterator, k):
        self.base_iterator = base_iterator
        self.k = k
    
    def __iter__(self):
        it = 0
        while it < self.k or self.k == -1:
            yield from self.base_iterator
            it += 1

    @staticmethod
    def __len__(self):
        return self.k


class TaskonomyDataModule(pl.LightningDataModule):
    '''
    PytorchLightning DataModule for Taskonomy.

    Args:
        taskonomy_root: Root directory of Taskonomy dataset (default: /datasets/taskonomy)
        taskonomy_domains: Choice of domain ID group: multiple specific IDs, separated by hyphen \
            (e.g. normal-depth_euclidean). (default: normal-depth_euclidean-reshading)
        taskonomy_variant: One of [full, fullplus, medium, tiny, debug] (default: fullplus)
        image_size: Input image size. (default: 256)
        batch_size: Batch size for data loader (default: 32)
        num_workers: Number of workers for DataLoader. (default: 16)
        pin_memory: Set to True to pin data loader memory (default: False)
        async_loader: Set to True to use asynchronous data loader. Only use with single GPU training. (default: False)
    '''
    def __init__(self, 
                 taskonomy_root: str = '/PATH_TO/taskonomy',
                 taskonomy_domains: str = 'normal-depth_euclidean-reshading',
                 taskonomy_variant: str = 'fullplus',
                 return_rgb: bool = True,
                 return_mask: bool = True,
                 image_size: int = 256,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 pin_memory: bool = False,
                 async_loader: bool = False,
                 max_images_train: int = None,
                 max_images_val: int = None,
                 max_images_test: int = None,
                 n_passes_epoch: int = 0,
                 cache: bool = False,
                 data_seed: int = -1,
                 rgb2lab: bool = False,
                 **kwargs):
        super().__init__()
        self.taskonomy_root = taskonomy_root
        self.taskonomy_domains = taskonomy_domains.split('-')
        self.taskonomy_variant = taskonomy_variant
        self.return_rgb = return_rgb
        self.return_mask = return_mask
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.async_loader = async_loader
        self.max_images_train = max_images_train
        self.max_images_val = max_images_val
        self.max_images_test = max_images_test
        self.kwargs = kwargs
        self.cache = cache
        self.n_passes_epoch = n_passes_epoch
        self.data_seed = data_seed
        self.rgb2lab = rgb2lab

    def setup(self, stage=None):
        ''' Creates Taskonomy datasets. '''
        print("Taskonomy variant: ", self.taskonomy_variant)
        
        load_domains = self.taskonomy_domains
        if self.return_mask:
            load_domains = load_domains if 'mask_valid' in load_domains else load_domains + ['mask_valid']
        if self.return_rgb:
            load_domains = load_domains if 'rgb' in load_domains else load_domains + ['rgb']

        if stage == 'fit' or stage is None:
            
            taskonomy_options_train = TaskonomyDataset.Options(
                data_path = self.taskonomy_root,
                tasks = load_domains,
                transform = 'DEFAULT',
                buildings = self.taskonomy_variant + '-train',
                image_size = self.image_size,
                max_images = self.max_images_train,
                shared_cache=Manager().dict(train_cache) if self.cache else None,
                data_seed=self.data_seed,
                rgb2lab=self.rgb2lab
            )
            taskonomy_options_val = TaskonomyDataset.Options(
                data_path = self.taskonomy_root,
                tasks = load_domains,
                transform = 'DEFAULT',
                buildings = self.taskonomy_variant + '-val',
                image_size = self.image_size,
                max_images = self.max_images_val,
                shared_cache=Manager().dict(val_cache) if self.cache else None,
                data_seed=self.data_seed,
                rgb2lab=self.rgb2lab
            )
            self.trainset = TaskonomyDataset(taskonomy_options_train)
            self.valset   = TaskonomyDataset(taskonomy_options_val)
            
        if stage == 'test' or stage is None:
            taskonomy_options_test = TaskonomyDataset.Options(
                data_path = self.taskonomy_root,
                tasks = load_domains,
                transform = 'DEFAULT',
                buildings = self.taskonomy_variant + '-test',
                image_size = self.image_size,
                max_images = self.max_images_test,
                rgb2lab=self.rgb2lab
            )
            self.testset  = TaskonomyDataset(taskonomy_options_test)

    def train_dataloader(self):
        ''' PyTorch Lightning method: Creates train set dataloader. '''
        sampler = KEpochsSampler(
            torch.utils.data.RandomSampler(self.trainset), self.n_passes_epoch,
        )
        loader = DataLoader(
            self.trainset, batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader

    def val_dataloader(self):
        ''' PyTorch Lightning method: Creates validation set dataloader. '''
        loader = DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader

    def test_dataloader(self):
        ''' PyTorch Lightning method: Creates test set dataloader. '''
        loader = DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader



class TaskonomyOnImagenetDataModule(pl.LightningDataModule):
    '''
    PytorchLightning DataModule for Taskonomy.

    Args:
        taskonomy_root: Root directory of Taskonomy dataset (default: /datasets/taskonomy)
        taskonomy_domains: Choice of domain ID group: multiple specific IDs, separated by hyphen \
            (e.g. normal-depth_euclidean). (default: normal-depth_euclidean-reshading)
        taskonomy_variant: One of [full, fullplus, medium, tiny, debug] (default: fullplus)
        image_size: Input image size. (default: 256)
        batch_size: Batch size for data loader (default: 32)
        num_workers: Number of workers for DataLoader. (default: 16)
        pin_memory: Set to True to pin data loader memory (default: False)
        async_loader: Set to True to use asynchronous data loader. Only use with single GPU training. (default: False)
    '''
    def __init__(self, 
                 taskonomy_root: str = '/PATH_TO/taskonomy',
                 taskonomy_domains: str = 'normal-depth_euclidean-reshading',
                 taskonomy_variant: str = 'all',
                 return_rgb: bool = True,
                 return_mask: bool = False,
                 image_size: int = 256,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 pin_memory: bool = False,
                 async_loader: bool = False,
                 **kwargs):
        super().__init__()
        self.taskonomy_root = taskonomy_root
        self.taskonomy_domains = taskonomy_domains.split('-')
        self.taskonomy_variant = taskonomy_variant
        self.return_rgb = return_rgb
        self.return_mask = return_mask
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.async_loader = async_loader
        self.kwargs = kwargs

    def setup(self, stage=None):
        ''' Creates Taskonomy datasets. '''
        load_domains = self.taskonomy_domains
        if self.return_mask:
            load_domains = load_domains if 'mask_valid' in load_domains else load_domains + ['mask_valid']
        if self.return_rgb:
            load_domains = load_domains if 'rgb' in load_domains else load_domains + ['rgb']
        
        if self.taskonomy_variant == 'all':
            paths = glob.glob(os.path.join(os.path.join(self.taskonomy_root, 'train', 'rgb'), "*", ""))
            self.taskonomy_variant = [p.split("/")[-2] for p in paths]
        if self.taskonomy_variant == 'debug':
            self.taskonomy_variant = ['n01440764']

        if stage == 'fit' or stage is None:
            taskonomy_options_train = TaskonomyDataset.Options(
                data_path = os.path.join(self.taskonomy_root, 'train'),
                tasks = load_domains,
                transform = 'DEFAULT',
                do_center_crop_transform = True,
                buildings = self.taskonomy_variant,
                image_size = self.image_size,
                file_name_parser = "NO_MULTIVIEW"
            )
            taskonomy_options_val = TaskonomyDataset.Options(
                data_path = os.path.join(self.taskonomy_root, 'val'),
                tasks = load_domains,
                transform = 'DEFAULT',
                do_center_crop_transform = True,
                buildings = self.taskonomy_variant,
                image_size = self.image_size,
                file_name_parser = "NO_MULTIVIEW"
            )
            self.trainset = TaskonomyDataset(taskonomy_options_train)
            self.valset   = TaskonomyDataset(taskonomy_options_val)
            
        if stage == 'test' or stage is None:
            taskonomy_options_test = TaskonomyDataset.Options(
                data_path = os.path.join(self.taskonomy_root, 'val'),
                tasks = load_domains,
                transform = 'DEFAULT',
                do_center_crop_transform = True,
                buildings = self.taskonomy_variant,
                image_size = self.image_size,
                file_name_parser = "NO_MULTIVIEW"
            )
            self.testset  = TaskonomyDataset(taskonomy_options_test)

    def train_dataloader(self):
        ''' PyTorch Lightning method: Creates train set dataloader. '''
        loader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader

    def val_dataloader(self):
        ''' PyTorch Lightning method: Creates validation set dataloader. '''
        loader = DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader

    def test_dataloader(self):
        ''' PyTorch Lightning method: Creates test set dataloader. '''
        loader = DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        if self.async_loader:
            loader = AsynchronousLoader(loader, q_size=self.num_workers)
        return loader