import torch
from abc import ABC, abstractmethod

class LinkModule(torch.nn.Module, ABC):
    '''
    This defines the API of a link module, connecting representation module
    outputs to down-stream task inputs.
    '''
    def __init__(self):
        super(LinkModule, self).__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

from .identity_link import *
from .convnet_activations_link import *