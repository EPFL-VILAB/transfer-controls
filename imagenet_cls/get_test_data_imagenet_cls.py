import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset.folder import ImageFolder

valdir = '/PATH_TO/imagenet/val'

"""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=64, 
    shuffle=False,
    num_workers=32, 
    pin_memory=True)
"""
test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
test_loader = torch.utils.data.DataLoader(
    ImageFolder(valdir, 
        split='test',
        transform=test_transform,
        rgb2lab=True,
    ),
    batch_size=64, 
    shuffle=False,
    num_workers=32, 
    pin_memory=True)

batches = []
for batch in tqdm(test_loader):
    batches.append(batch)

print("===> Saving batches")
torch.save(batches, f'PATH_TO/test_batches_imgnet_cls_color_50k.torch')