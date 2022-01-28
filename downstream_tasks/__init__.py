import os
import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class DownstreamModule(pl.LightningModule, ABC):
    '''
    This defines the API of a down-stream task module.
    '''
    def __init__(self):
        super(DownstreamModule, self).__init__()
        self.save_debug_info_on_error = False

    @abstractmethod
    def _shared_step(self, batch, is_train):
        raise NotImplementedError()

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

from .taskonomy_dst import *