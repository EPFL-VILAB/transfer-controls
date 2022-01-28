import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from PIL import Image

from . import DownstreamModule
from representation_tasks import RepresentationModule
from models import UNet, UNetDecoder, ConstantPredictor
from data import taskonomy_task_configs
from utils import make_valid_mask, masked_l1_loss, ensure_valid_mask, masked_l1_loss_batches

# import rgb2lab transform, for log ood images transforms.
# from data.taskonomy.transforms import convertrgb2lab

import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class TaskonomyDownstreamModule(DownstreamModule):
    def __init__(self,
                 representation_module,
                 link_module,
                 freeze_representation,
                 taskonomy_domain,
                 model_name,
                 in_channels,
                 image_size,
                 lr,
                 weight_decay=1e-2,
                 lr_step=None,
                 lr_warmup_step=None,
                 valset=None,
                 testset=None,
                 layer_ids=None,
                 x2y=False,
                 **kwargs):
        super(TaskonomyDownstreamModule, self).__init__()
        self.representation_module = representation_module
        self.link_module = link_module
        self.freeze_representation = freeze_representation
        if freeze_representation:
            self.representation_module.freeze()
        self.taskonomy_domain = taskonomy_domain
        self.model_name = model_name
        self.in_channels = in_channels
        self.image_size = image_size
        self.lr = lr
        self.lr_step = lr_step
        self.lr_warmup_step = lr_warmup_step
        self.valset = valset
        self.testset = testset
        self.layer_ids = layer_ids
        self.x2y = x2y
        self.gpus = kwargs['gpus']
        self.kwargs = kwargs
        self.save_hyperparameters(
            'freeze_representation', 'taskonomy_domain', 'model_name', 
            'in_channels', 'image_size', 'lr', 'lr_step', 'weight_decay',
            *[k for k,v in kwargs.items() if isinstance(v,(str, int, float, bool)) or v is None]
        )

        self.test_losses = []

        self.setup_model()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--freeze_representation', default=False, action='store_true',
            help='Set to freeze representation network. (default: False)')
        parser.add_argument(
            '--taskonomy_domain', type=str, default='normal',
            help='Choice of single Taskonomy domain. (default: normal)')
        parser.add_argument(
            '--model_name', type=str, default='unet',
            help='Model type. (default: unet)')
        parser.add_argument(
            '--in_channels', type=int, default=3,
            help='Input channels. (default: 3)')
        parser.add_argument(
            '--image_size', type=int, default=256,
            help='Input image size. (default: 256)')
        parser.add_argument(
            '--lr', type=float, default=1e-3,
            help='Learning rate. (default: 1e-3)')
        parser.add_argument(
            '--weight_decay', type=float, default=1e-2,
            help='Weight decay. (default: 1e-2)')
        parser.add_argument(
            '--lr_step', type=int, default=None,
            help='Learning rate scheduler step (epoch). Not used if None. (default: None)')
        parser.add_argument(
            '--lr_warmup_step', type=int, default=0,
            help='Learning rate warmup step (global_step). Not used if 0. (default: None)')
        parser.add_argument(
            '--valset', default=None,
            help='Optional validation set for logging example images to W&B. (default: None)')
        parser.add_argument(
            '--lr_decay', type=str, default='none'
        )
        parser.add_argument(
            '--optimizer', type=str, default='adamw'
        )
        return parser

    def setup_model(self):
        """create encoder and decoder models"""
        if self.x2y:
            print("Using another UNet model for X->Y.")
            if self.taskonomy_domain == 'normal':
                # from depth-> normal
                out_channels = 1        # out_channels for the mid-prediction
            elif self.taskonomy_domain == 'depth_zbuffer':
                # from normal->depth
                out_channels = 3
        else:
            out_channels = taskonomy_task_configs.task_parameters[self.taskonomy_domain]['out_channels']
            #out_channels = taskonomy_task_configs.task_parameters[self.taskonomy_domain]['num_channels']
        
        if 'unet_decoder_skip' in self.model_name:
            if self.model_name == 'unet_decoder_skip':
                # Standard 6 layer UNet decoder
                downstream_model = UNetDecoder(upsample=6, out_channels=out_channels)
            elif self.model_name.split('_')[-1].isnumeric():
                # unet_decoder_skip_X -> X layer UNet decoder
                upsample = int(self.model_name.split('_')[-1])
                downstream_model = UNetDecoder(upsample=upsample, out_channels=out_channels)

        elif 'unet' in self.model_name:
            if self.model_name == 'unet':
                # Standard 6 layer UNet
                downstream_model = UNet(downsample=6, in_channels=self.in_channels, out_channels=out_channels)
            elif self.model_name.split('_')[-1].isnumeric():
                # unet_X -> X down and X up layer UNet
                downsample = int(self.model_name.split('_')[-1])
                downstream_model = UNet(downsample=downsample, in_channels=self.in_channels, out_channels=out_channels)

        elif self.model_name == 'constantpredictor':
            downstream_model = ConstantPredictor((out_channels, self.image_size, self.image_size))
        
        else:
            # TODO: Implement other models that deal with different input representation shapes
            raise NotImplementedError()

        # Add a link module
        if self.link_module is not None:
            self.model = nn.Sequential(self.link_module, downstream_model)
        else:
            self.model = downstream_model

    def forward(self, x):
        '''
        Runs forward pass through the representation module and network for given RGB input.

        Args:
            x: Input RGB image of shape (batch x 3 x height x width).
        Returns:
            Prediction for chosen task.
        '''
        with torch.no_grad() if self.freeze_representation else dummy_context_mgr():

            if self.freeze_representation:
                self.representation_module.eval()
                
            representation = self.representation_module.get_representation(x, layer_ids=self.layer_ids)
            if isinstance(representation, dict):
                representation['input'] = x

        preds = self.model(representation)
        if self.x2y:
            x2y_preds = self.unet(preds)
            preds = x2y_preds

        return preds

    def training_step(self, batch, batch_idx):
        ''' PyTorch Lightning method: Runs and logs one training step. '''
        loss = self.shared_step(batch, is_train=True)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=self.gpus>1)

        # log the current lr
        opt = self.optimizers()
        cur_lr = None
        for param_group in opt.param_groups:
            cur_lr = param_group['lr']

        self.log('traing lr', cur_lr, prog_bar=True, logger=True, sync_dist=self.gpus>1)
        
        return loss

    def validation_step(self, batch, batch_idx):
        ''' PyTorch Lightning method: Runs and logs one validation step. '''
        loss = self.shared_step(batch, is_train=False)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=self.gpus>1)
        
        return loss


    #========================================================
    # Add for test
    # trainer.test(data_module)
    def test_step(self, batch, batch_idx):
        ''' PyTorch Lightning method: Runs and logs one test step. '''
        loss = self.shared_step(batch, is_train=False)

        # Logging
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=self.gpus>1)
        
        return loss
    #========================================================


    def _shared_step(self, batch, is_train=True):
        rgb = batch['rgb']
        mask_valid = ensure_valid_mask(batch, self.image_size, if_missing_return_ones_like=batch[self.taskonomy_domain])
        
        # Compute model(link_module(representation(input)))
        pred = self(rgb)
        
        if self.taskonomy_domain == 'principal_curvature':
            pred = pred[:, :2, :, :]

        loss = masked_l1_loss(
            pred,
            batch[self.taskonomy_domain], 
            mask_valid.repeat_interleave(pred.shape[1],1)
        )

        return loss
 
    def validation_epoch_end(self, outputs):
        ''' Runs after validation epoch. Used for logging example images. '''
        if self.global_rank > 0:
            return 
        # Log validation set and OOD debug images using W&B
        self.log_validation_example_images(num_images=10)

        # comment log ood images when using rgb2lab convert, since the default transform is not applicable. 
        # self.log_ood_example_images(num_images=10)


    def log_validation_example_images(self, num_images=10, seed=0):
        ''' Logs validation set image predictions on W&B. '''
        if self.valset is None:
            return

        self.eval()

        all_imgs = {'val_rgb': [], 'val_pred': [], 'val_gt': []}

        np.random.seed(seed)

        if len(self.valset) < num_images:
            num_images = len(self.valset)
        img_idxs = np.random.choice(np.arange(len(self.valset)), num_images, replace=False)

        for i, img_idx in enumerate(img_idxs):
            example = self.valset[img_idx]
            rgb = example['rgb'].to(self.device)
            gt = example[self.taskonomy_domain]
            
            if 'mask_valid' in example:
                example['mask_valid'] = example['mask_valid'].unsqueeze(0)
            mask_valid = ensure_valid_mask(example, self.image_size, gt.unsqueeze(0))[0]

            with torch.no_grad():
                pred = self(rgb.unsqueeze(0))[0]

            rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb, caption=f'RGB {i}')
            all_imgs['val_rgb'].append(rgb)

            pred[~mask_valid.repeat_interleave(pred.shape[0],0)] = 0
            pred = wandb.Image(pred, caption=f'Pred {i}')
            all_imgs['val_pred'].append(pred)

            gt[~mask_valid.repeat_interleave(gt.shape[0],0)] = 0
            gt = wandb.Image(gt, caption=f'GT {i}')
            all_imgs['val_gt'].append(gt)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    #==========================================================================
    # log some test images
    def test_epoch_end(self, outputs):
        if self.global_rank > 0:
            return 
        
        # save test images
        self.log_test_example_images(num_images=10)


    def log_test_example_images(self, num_images=10, seed=0):
        ''' Logs test set image predictions on W&B. '''
        if self.testset is None:
            return

        self.eval()

        all_imgs = {'test_rgb': [], 'test_pred': [], 'test_gt': []}

        np.random.seed(seed)
        img_idxs = np.random.choice(np.arange(len(self.testset)), num_images, replace=False)

        for i, img_idx in enumerate(img_idxs):
            example = self.testset[img_idx]
            rgb = example['rgb'].to(self.device)
            gt = example[self.taskonomy_domain]
            
            if 'mask_valid' in example:
                example['mask_valid'] = example['mask_valid'].unsqueeze(0)
            mask_valid = ensure_valid_mask(example, self.image_size, gt.unsqueeze(0))[0]

            with torch.no_grad():
                pred = self(rgb.unsqueeze(0))[0]

            rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb, caption=f'RGB {i}')
            all_imgs['test_rgb'].append(rgb)

            pred[~mask_valid.repeat_interleave(pred.shape[0],0)] = 0
            pred = wandb.Image(pred, caption=f'Pred {i}')
            all_imgs['test_pred'].append(pred)

            gt[~mask_valid.repeat_interleave(gt.shape[0],0)] = 0
            gt = wandb.Image(gt, caption=f'GT {i}')
            all_imgs['test_gt'].append(gt)

        self.logger.experiment.log(all_imgs, step=self.global_step)
    #============================================================================


    def log_ood_example_images(self, data_dir='/datasets/evaluation_ood/real_world/images', num_images=10):
        ''' Logs out-of-distribution image predictions on W&B. '''
        self.eval()

        all_imgs = {'rgb_ood': [], 'pred_ood': []}

        for img_idx in range(num_images):
            rgb = Image.open(f'{data_dir}/{img_idx:05d}.png').convert('RGB')
            rgb = self.valset.transform['rgb'](rgb).to(self.device)

            with torch.no_grad():
                pred = self(rgb.unsqueeze(0))[0]

            rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb = wandb.Image(rgb, caption=f'RGB OOD {img_idx}')
            all_imgs['rgb_ood'].append(rgb)

            pred = wandb.Image(pred, caption=f'Pred OOD {img_idx}')
            all_imgs['pred_ood'].append(pred)

        self.logger.experiment.log(all_imgs, step=self.global_step)

    def configure_optimizers(self):
        ''' PyTorch Lightning method: Sets up optimizer and scheduler. '''
        if self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.hparams.weight_decay,
                amsgrad=False
            )

        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hparams.weight_decay,
            )

        print(optimizer)
        if self.lr_step is not None:
           scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step)
           return [optimizer], [scheduler]
        else:
           return optimizer


    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #     scheduler_dict = {
    #         'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9),
    #         'interval': 'step',  # called after each training step
    #         "frequency": 5000,
    #         }
    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        ''' PyTorch Lightning method: Performs an optimizer step with LR warmup. '''
        # TODO: implement via scheduler
        # Warm up learning rate
        if self.trainer.global_step < self.lr_warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.lr_warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        elif self.hparams.lr_decay == 'linear':
            lr_scale = 1. - min(1., float(self.trainer.global_step - self.lr_warmup_step + 1) / float(self.hparams.max_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        elif self.hparams.lr_decay == 'cosine':
            cur_step = self.trainer.global_step
            max_step = self.hparams.max_steps
            rate = cur_step / max_step
            lr_scale = (1 + np.cos(rate)) / 2
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
class TaskonomyClassificationModule(TaskonomyDownstreamModule):
    def __init__(self, n_classes, *args, **kwargs):
        self.n_classes = n_classes
        self.tau = 1.0
        super(TaskonomyClassificationModule, self).__init__(*args, **kwargs)

    def setup_model(self):
        """create encoder and decoder models"""
        downstream_model = nn.Linear(2048, self.n_classes)

        # Add a link module
        if self.link_module is not None:
            self.model = nn.Sequential(self.link_module, downstream_model)
        else:
            self.model = downstream_model

    def forward(self, x):
        '''
        Runs forward pass through the representation module and network for given RGB input.

        Args:
            x: Input RGB image of shape (batch x 3 x height x width).
        Returns:
            Prediction for chosen task.
        '''
        if self.freeze_representation:
            self.representation_module.eval()

        with torch.no_grad() if self.freeze_representation else dummy_context_mgr():
            if self.freeze_representation:
                self.representation_module.eval()
            representation = self.representation_module.get_representation(x, layer_ids=self.layer_ids)
        preds = self.model(representation['avgpool'])

        return preds / self.tau

    def training_step(self, batch, batch_idx):
        ''' PyTorch Lightning method: Runs and logs one training step. '''
        loss, acc = self.shared_step(batch, is_train=True)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=self.gpus>1)
        self.log('train_acc', acc, logger=True, sync_dist=self.gpus>1)
        
        # log the current lr
        opt = self.optimizers()
        cur_lr = None
        for param_group in opt.param_groups:
            cur_lr = param_group['lr']

        self.log('traing lr', cur_lr, logger=True, sync_dist=self.gpus>1)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        ''' PyTorch Lightning method: Runs and logs one validation step. '''
        if dataloader_idx != 1:
            loss, acc = self.shared_step(batch, is_train=False)
            
            # Logging
            self.log('val_loss', loss, logger=True, sync_dist=self.gpus>1)
            self.log('val_acc', acc, logger=True, sync_dist=self.gpus>1)
        else:
            loss = self.test_step(batch, batch_idx)
        return loss


    #========================================================
    # Add for test
    # trainer.test(data_module)
    def test_step(self, batch, batch_idx):
        ''' PyTorch Lightning method: Runs and logs one test step. '''
        loss, acc = self.shared_step(batch, is_train=False)

        # Logging
        self.log('test_loss', loss, logger=True, sync_dist=self.gpus>1)
        self.log('test_acc', acc, logger=True, sync_dist=self.gpus>1)
        return loss
    #========================================================


    def _shared_step(self, batch, is_train=True):
        rgb = batch['rgb']

        # Compute model(link_module(representation(input)))
        pred = self(rgb)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, batch[self.taskonomy_domain])
        acc = (pred.argmax(1) == batch[self.taskonomy_domain]).float().mean()
        return loss, acc
    
    def validation_epoch_end(self, outputs):
        pass
    
    def test_epoch_end(self, outputs):
        pass
