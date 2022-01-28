import os
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class RepresentationModule(pl.LightningModule, ABC):
    '''
    This defines the API of a representation module.
    '''
    def __init__(self):
        super(RepresentationModule, self).__init__()
        self.save_debug_info_on_error = False

    @abstractmethod
    def get_representation(self, x, layer_id: str = None):
        raise NotImplementedError()

    @abstractmethod
    def _shared_step(self, batch, is_train):
        raise NotImplementedError()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def register_save_on_error_callback(self, callback):
        '''
            On error, will call the following callback. 
            Callback should have signature:
                callback(batch) -> none
        '''
        self.on_error_callback = callback
        self.save_debug_info_on_error = True
        
    def shared_step(self, batch, is_train=True):
        try:
            return self._shared_step(batch, is_train)
        except:
            if self.save_debug_info_on_error:
                self.on_error_callback(batch)
            raise

    def save_model_and_batch_on_error(self, checkpoint_function, save_path_prefix='.'):
        def _save(batch):
            checkpoint_function(os.path.join(save_path_prefix, "crash_model.pth"))
            print(f"Saving crash information to {save_path_prefix}")
            with open(os.path.join(save_path_prefix, "crash_batch.pth"), 'wb') as f:
                torch.save(batch, f)
        return _save


class IdentityRepresentation(RepresentationModule):
    def get_representation(self, x, **kwargs):
        return x

    def _shared_step(self, batch, is_train):
        raise NotImplementedError()


from .rgb_module import *
from .moco_module import *
from .swav_module import *
from .simclr_module import *
from .barlow_twins import *
from .resnet_scratch import *
from .imagenet_module import *
from .moco_virb_module import *
from .swav_virb_module import *
from .pirl_module import *
from .simsiam_module import *
from .colorization_encoder import *
from .taskonomy_network import *
from .colorization_taskonomy_encoder import *
from .jigsaw_module import *
from .jigsaw_taskonomy_encoder import *
from .task_best_encoder import *

