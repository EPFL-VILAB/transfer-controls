from typing import Iterator, Dict, Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import os

from . import RepresentationModule
from models import *
from utils import FeatureExtractor, init_conv_dct


class Colorization(RepresentationModule):
    """
    Module that return GRAY -> Colorization pretext task trained on ImageNet features

    Args:
        pretrained_weights_path: Optional string to load locally stored weights
    """
    def __init__(self, pretrained_weights_path, size, pretrained=True):
        super(Colorization, self).__init__()
        self.pretrained_weights_path = pretrained_weights_path
        self.size = size

        self.model = self.setup_model()

        if pretrained:
            if os.path.isfile(self.pretrained_weights_path):
                print("=> loading checkpoint '{}'".format(self.pretrained_weights_path))

                # load VISSL pretrained model
                if torch.cuda.is_available():
                    state_dict = torch.load(self.pretrained_weights_path)
                else:
                    state_dict = torch.load(self.pretrained_weights_path, map_location=torch.device('cpu'))
    
                #print(state_dict['model_state_dict'].keys())                
                state_dict = state_dict['model_state_dict']

                # remove prefixe "_feature_blocks."
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('_feature_blocks.'):
                        # remove prefix
                        state_dict[k[len("_feature_blocks."):]] = state_dict[k]
    
                    # delete renamed or unused k
                    del state_dict[k]

                msg = self.model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(self.pretrained_weights_path))
            else:
                print("=> no checkpoint found at '{}'".format(self.pretrained_weights_path))


    def setup_model(self):
        model = torchvision.models.resnet50(pretrained=False)
        # modify the conv1 input as 1 channel for gray scale.
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
