from typing import Iterator, Dict, Union, Callable, Optional
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import os

from . import RepresentationModule
from models import *
from utils import FeatureExtractor, init_conv_dct


class ResNetScratch(RepresentationModule):
    """
    Module that return RGB -> ResNet50 features

    Args:
        pretrained_weights_path: Optional string to load locally stored weights
    """
    def __init__(self, size, pretrained=False):
        super().__init__()
        self.size = size
        self.model = self.setup_model()

    def setup_model(self):
        model = torchvision.models.resnet50(pretrained=False)
        return model

    def get_representation(self, x: torch.Tensor, layer_ids: Union[str, Iterable[str], None] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        #x = TF.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        if self.size is not None:
            x = F.interpolate(x, size=self.size)

        if layer_ids is None:
            return self(x)
        if 'feature_extractor' not in dir(self):
            self.feature_extractor = FeatureExtractor(self.model, layer_ids)
        return self.feature_extractor.forward(x)


    def forward(self, x):
        #x = TF.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        _, _, h, w = x.shape
        if self.size is not None:
            x = F.interpolate(x, size=224)  
        
        preds = self.model(x)

        if self.size is not None:
            preds = F.interpolate(preds, size=(h,w))
        
        return preds

    def _shared_step(self, batch, is_train):
        # TODO: Optional. Add code to train / continue training on ImageNet
        pass


if __name__ == "__main__":
    # Prepare Representation Module
    representation_module = ResNetScratch()