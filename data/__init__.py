from .taskonomy.taskonomy_dataset import TaskonomyDataset, TaskonomyDataLoader
from .taskonomy.taskonomy_datamodule import TaskonomyDataModule, TaskonomyOnImagenetDataModule
from .taskonomy import task_configs as taskonomy_task_configs

from .asynchronous_loader import AsynchronousLoader
from .cifar100 import CIFAR100Dataset, CIFAR100DataModule
from .eurosat import EurosatDataset, EurosatDataModule
from .clevr import CLEVRDataset, CLEVRDataModule