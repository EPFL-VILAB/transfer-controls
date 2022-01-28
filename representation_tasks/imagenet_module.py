from typing import Iterator, Dict, Union, Callable, Optional
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import os

from . import RepresentationModule, ResNetScratch
from models import *
from utils import FeatureExtractor, init_conv_dct


class ResNetImageNetPretrained(ResNetScratch):
    """
    Module that return RGB -> ResNet50 ImageNet pretrained features

    Args:
        pretrained_weights_path: Optional string to load locally stored weights
    """
    def setup_model(self):
        model = torchvision.models.resnet50(pretrained=True)
        return model

if __name__ == "__main__":
    # Prepare Representation Module
    representation_module = ResNetImageNetPretrained()