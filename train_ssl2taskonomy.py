# used to transfer training the imagenet pre-trained SSL encoders to
# downstream tasks, like normal, depth_zbuffer, reshading.

import os
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
import random

from representation_tasks import IdentityRepresentation, MoCoModule, SwAVModule, SimCLRModule, BarlowTwins, ResNetScratch, ResNetImageNetPretrained
from representation_tasks import MoCo_ViRB_Module, SwAV_ViRB_Module
from representation_tasks import IdentityRepresentation, Colorization, ColorizationTask, Jigsaw, JigsawTask
from representation_tasks import TaskBestEncoder
from representation_tasks import PIRLModule, SimSiamModule

from link_modules import ConvNetActivationsLink, IdentityLink
from downstream_tasks import TaskonomyDownstreamModule, TaskonomyClassificationModule
from data import TaskonomyDataModule, CIFAR100DataModule, CLEVRDataModule, EurosatDataModule

model_paths = {
    'moco': 'PATH_TO/moco_v2_800ep_pretrain.pth.tar',
    'swav': 'PATH_TO/swav_800ep_pretrain.pth.tar',
    'simclr': 'PATH_TO/simclr_800ep.torch',
    'barlow': 'PATH_TO/barlow_twins_1000ep.pth',
    'moco_task': 'PATH_TO/MoCov2Taskonomy.pt',
    'swav_task': 'PATH_TO/SWAVTaskonomy.pt',
    'colorization': 'PATH_TO/vissl_colorization_rn50.torch',
    'colorization_task': 'PATH_TO/task_colorization_encoder.dat',
    'jigsaw': 'PATH_TO/vissl_jigsaw_in1k_rn50.torch',
    'jigsaw_task': 'PATH_TO/task_jigsaw_encoder.dat',
    'normal': 'PATH_TO/normal_best.ckpt',
    'reshading': 'PATH_TO/reshading_best.ckpt',
    'simsiam': 'PATH_TO/simsiam_100ep_256bs.pth.tar',
    'pirl': 'PATH_TO/pirl_800ep_liner.torch'
}

if __name__ == "__main__":
    # Experimental setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ssl_name', type=str, default='moco',
        help='One of [moco, swav, simclr, barlow, scratch, sup, moco_task, swav_task] (default: moco)')
    parser.add_argument(
        '--resnet_name', type=str, default='resnet50',
        help='torchvision ResNe(x)t identifier (default: resnet50)')
    parser.add_argument(
        '--link_model_name', type=str, default='convnet',
        help='Link module')
    parser.add_argument(
        '--downstream_model', type=str, default='unet_decoder_skip_6',
        help='Decoder model')
    parser.add_argument(
        '--pretrained_weights_path', type=str, default=None,
        help='Path to pretrained weights (default: None)')
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name for Weights & Biases. (default: None)')
    parser.add_argument(
        '--restore', type=str, default=None,
        help='Weights & Biases ID to restore and resume training. (default: None)')
    parser.add_argument(
        '--save-on-error', type=bool, default=True,
        help='Save crash information on fatal error. (default: True)')    
    parser.add_argument(
        '--save-dir', type=str, default='exps',
        help='Directory in which to save this experiments. (default: exps/)')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for data loader (default: 64)')
    parser.add_argument(
        '--num_workers', type=int, default=32,
        help='Number of workers for DataLoader. (default: 32)')
    parser.add_argument(
        '--n_passes_epoch', type=int, default=1,
        help='<n_passes_epoch> passes over dataset is made in one epoch instead of 1. Infinite number of passes if -1 (default: 1)')
    #parser.add_argument(
    #    '--cache', default=False, action='store_true',
    #    help='Set to pin memory. (default: False)')
    parser.add_argument(
        '--pin_memory', default=False, action='store_true',
        help='Set to pin memory. (default: False)')
    parser.add_argument(
        '--max_images_train', type=int, default=None,
        help='Number of training images. Uses all if None. (default: None)')
    parser.add_argument(
        '--max_images_val', type=int, default=None,
        help='Number of validation images. Uses all if None. (default: None)')
    parser.add_argument(
        '--max_images_test', type=int, default=None,
        help='Number of testing images. Uses all if None. (default: None)')
    parser.add_argument(
        '--ckpt_period', type=int, default=1,
        help='Interval between checkpoints. (default: 1)')
    parser.add_argument(
        '--taskonomy_variant', type=str, default='fullplus',
        choices=['full', 'fullplus', 'medium', 'tiny', 'debug'],
        help='One of [full, fullplus, medium, tiny, debug] (default: fullplus)')
    parser.add_argument(
        '--taskonomy_root', type=str, default='/datasets/taskonomy',
        help='Root directory of Taskonomy dataset (default: /datasets/taskonomy)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Global seed')
    parser.add_argument(
        '--data_seed', type=int, default=0,
        help='Seed for dataset splitting')
    parser.add_argument(
        '--rgb2lab', default=False, action='store_true',
        help='Convert RGB to Lab. (default: False)')
    parser.add_argument(
        '--dataset', default='taskonomy'
    )
    parser.add_argument(
        '--tmp', default=False, action='store_true',
        help='Tmp run. (default: False)')
    parser.add_argument(
        '--version', type=str, default=None,
        help='Suffix append to the exp name (default: None)')
    parser.add_argument(
        '--stratified', default=False, action='store_true',)

    # Add PyTorch Lightning Module and Trainer args
    parser = TaskonomyDownstreamModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=torch.cuda.device_count())
    args = parser.parse_args()

    # Set data_seed and setup dataset
    # Prepare Taskonomy datasets
    torch.manual_seed(args.data_seed)
    torch.cuda.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    print("Preparing Taskonomy datasets ...")
    args.taskonomy_domains = args.taskonomy_domain
    print("Taskonomy domains: ", args.taskonomy_domains)
    
    if args.dataset == 'taskonomy':
        data_module = TaskonomyDataModule(**vars(args))    
    elif args.dataset == 'cifar100':
        data_module = CIFAR100DataModule(**vars(args))
    elif args.dataset == 'eurosat':
        data_module = EurosatDataModule(**vars(args))
    elif args.dataset == 'clevr':
        data_module = CLEVRDataModule(**vars(args))
        
    data_module.setup()
    args.valset = data_module.valset
    # add testset to taskonomy_dst_module to plot some test figs.
    args.testset = data_module.testset

    # Set all the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.experiment_name is None:
        args.experiment_name = args.ssl_name + f'_{args.taskonomy_domain}'
        if args.max_images_train:
            args.experiment_name += f'_{args.max_images_train/1000}K'
        if args.freeze_representation:
            args.experiment_name += '_frozen'

        if args.version is not None:
            args.experiment_name += f'_{args.version}'

    if args.tmp:
        args.experiment_name = f'tmp-{args.experiment_name}'


    # Prepare Representation Module
    print("Preparing Representation Module ...")
    if args.ssl_name == 'moco':
        representation_module = MoCoModule(
            pretrained_weights_path=model_paths['moco'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'swav':
        representation_module = SwAVModule(
            pretrained_weights_path=model_paths['swav'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'simclr':
        representation_module = SimCLRModule(
            pretrained_weights_path=model_paths['simclr'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'barlow':
        representation_module = BarlowTwins(
            pretrained_weights_path=model_paths['barlow'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'simsiam':
        representation_module = SimSiamModule(
            pretrained_weights_path=model_paths['simsiam'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'pirl':
        representation_module = PIRLModule(
            pretrained_weights_path=model_paths['pirl'],
            size=224 if 'clip_' in args.resnet_name else None
        )

    elif args.ssl_name == 'moco_task':
        representation_module = MoCo_ViRB_Module(
            pretrained_weights_path=model_paths['moco_task'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'swav_task':
        representation_module = SwAV_ViRB_Module(
            pretrained_weights_path=model_paths['swav_task'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'color':
        representation_module = Colorization(
            pretrained_weights_path=model_paths['colorization'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'color_task':
        representation_module = ColorizationTask(
            pretrained_weights_path=model_paths['colorization_task'],
            size=224 if 'clip_' in args.resnet_name else None
        )

    elif args.ssl_name == 'jigsaw':
        representation_module = Jigsaw(
            pretrained_weights_path=model_paths['jigsaw'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'jigsaw_task':
        representation_module = JigsawTask(
            pretrained_weights_path=model_paths['jigsaw_task'],
            size=224 if 'clip_' in args.resnet_name else None
        )
        
    # these are not ssl method, but best scratch model pretrained for transferring
    elif args.ssl_name == 'normal':
        representation_module = TaskBestEncoder(
            pretrained_weights_path=model_paths['normal'],
            size=224 if 'clip_' in args.resnet_name else None
        )
    elif args.ssl_name == 'reshading':
        representation_module = TaskBestEncoder(
            pretrained_weights_path=model_paths['reshading'],
            size=224 if 'clip_' in args.resnet_name else None
        )

    elif args.ssl_name == 'scratch':
        representation_module = ResNetScratch(size=224 if 'clip_' in args.resnet_name else None)
    elif args.ssl_name == 'sup':
        representation_module = ResNetImageNetPretrained(size=224 if 'clip_' in args.resnet_name else None)
    elif args.ssl_name == 'blind-guess':
        representation_module = IdentityRepresentation()

    # Prepare Link Module
    print("Preparing Link Module ...")
    if args.ssl_name == 'jigsaw_task' or args.ssl_name == 'color_task':
        layer_map = [
            #{'src': 'input', 'c_in': 3, 'c_out': 16, 'k': 1, 's': 1, 'p': 0},
            {'src': 'conv1', 'c_in': 64, 'c_out': 16, 'k': 1, 's': 1, 'p': 0, 'size': 256},
            {'src': 'conv1', 'c_in': 64, 'c_out': 32, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer1', 'c_in': 256, 'c_out': 64, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer2', 'c_in': 512, 'c_out': 128, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer3', 'c_in': 1024, 'c_out': 256, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer4', 'c_in': 2048, 'c_out': 512, 'k': 1, 's': 2, 'p': 0},    # set s=2 here, to reduce the final feature map size
            {'src': 'layer4', 'c_in': 2048, 'c_out': 1024, 'k': 7, 's': 3, 'p': 0}
        ]
    elif args.resnet_name in ['resnet18', 'resnet34']:
        layer_map = [
            #{'src': 'input', 'c_in': 3, 'c_out': 16, 'k': 1, 's': 1, 'p': 0},
            {'src': 'conv1', 'c_in': 64, 'c_out': 16, 'k': 1, 's': 1, 'p': 0, 'size': 256},
            {'src': 'conv1', 'c_in': 64, 'c_out': 32, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer1'}, {'src': 'layer2'}, {'src': 'layer3'}, {'src': 'layer4'},
            {'src': 'layer4', 'c_in': 512, 'c_out': 1024, 'k': 7, 's': 2, 'p': 3},
        ]
    elif args.resnet_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_dct']:
        layer_map = [
            #{'src': 'input', 'c_in': 3, 'c_out': 16, 'k': 1, 's': 1, 'p': 0},
            {'src': 'conv1', 'c_in': 64, 'c_out': 16, 'k': 1, 's': 1, 'p': 0, 'size': 256},
            {'src': 'conv1', 'c_in': 64, 'c_out': 32, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer1', 'c_in': 256, 'c_out': 64, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer2', 'c_in': 512, 'c_out': 128, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer3', 'c_in': 1024, 'c_out': 256, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer4', 'c_in': 2048, 'c_out': 512, 'k': 1, 's': 1, 'p': 0},
            {'src': 'layer4', 'c_in': 2048, 'c_out': 1024, 'k': 7, 's': 2, 'p': 3}
        ]
    elif args.resnet_name in ['clip_RN50']:
        layer_map = [
            #{'src': 'input', 'c_in': 3, 'c_out': 16, 'k': 1, 's': 1, 'p': 0, 'size': 256},
            {'src': 'conv1', 'c_in': 32, 'c_out': 16, 'k': 1, 's': 1, 'p': 0, 'size': 256},
            {'src': 'conv1', 'c_in': 32, 'c_out': 32, 'k': 1, 's': 1, 'p': 0, 'size': 128},
            {'src': 'layer1', 'c_in': 256, 'c_out': 64, 'k': 1, 's': 1, 'p': 0, 'size': 64},
            {'src': 'layer2', 'c_in': 512, 'c_out': 128, 'k': 1, 's': 1, 'p': 0, 'size': 32},
            {'src': 'layer3', 'c_in': 1024, 'c_out': 256, 'k': 1, 's': 1, 'p': 0, 'size': 16},
            {'src': 'layer4', 'c_in': 2048, 'c_out': 512, 'k': 1, 's': 1, 'p': 0, 'size': 8},
            {'src': 'layer4', 'c_in': 2048, 'c_out': 1024, 'k': 7, 's': 2, 'p': 3, 'size': 4}
        ]
    
    if args.ssl_name == 'blind-guess':
        link_module = IdentityLink()
    elif args.taskonomy_domain == 'classification':
        link_module = torch.nn.Flatten()
    else:
        link_module = ConvNetActivationsLink(layer_map)

    # Extract activations of these resnet layers
    resnet_layer_ids = [
        'conv1', 'layer1', 'layer2', 'layer3', 'layer4'
    ]

    # Prepare Downstream Module
    print("Preparing Downstream Module ...")
    if args.taskonomy_domain == 'classification':
        if args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'eurosat':
            n_classes = 10
        elif args.dataset == 'clevr':
            n_classes = 11
            
        model = TaskonomyClassificationModule(
            n_classes=n_classes,
            representation_module=representation_module,
            link_module=link_module,
            model_name=None,
            layer_ids=['avgpool'],
            **{k:v for k,v in vars(args).items() if k not in ['model_name']}
        )
        
    else:
        model = TaskonomyDownstreamModule(
            representation_module=representation_module,
            link_module=link_module,
            model_name=args.downstream_model,
            layer_ids=resnet_layer_ids,
            **{k:v for k,v in vars(args).items() if k not in ['model_name']}
        )

    os.makedirs(os.path.join(args.save_dir, 'wandb'), exist_ok=True)
    wandb_logger = WandbLogger(
        name=args.experiment_name,
        project='{PROJECT-NAME}',
        entity='YOUR-ENTITY',
        save_dir = args.save_dir,
        version=args.restore
    )
    wandb_logger.watch(model, log=None, log_freq=1)

    # Save best and last model like ./checkpoints/taskonomy_representations/W&BID/epoch-X.ckpt (or .../last.ckpt)
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints', f'{wandb_logger.name}', f'{wandb_logger.experiment.id}')
    checkpoint_callback = ModelCheckpoint(verbose=True, monitor='val_loss', mode='min', period=args.ckpt_period, save_last=True)

    if args.restore is None:
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, checkpoint_callback=checkpoint_callback)
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=os.path.join(checkpoint_dir, 'last.ckpt'), 
            logger=wandb_logger, checkpoint_callback=checkpoint_callback
        )

    if args.save_on_error:
        model.register_save_on_error_callback(
            model.save_model_and_batch_on_error(
                trainer.save_checkpoint,
                args.save_dir
            )
        )
    
    print("Start training ...")
    trainer.fit(model, data_module)

    ## TODO Test
    trainer.test(datamodule=data_module)
