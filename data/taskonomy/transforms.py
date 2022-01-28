from PIL import Image, ImageCms
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from   typing import Optional
from scipy import ndimage

from . import task_configs

try:
    import accimage
except ImportError:
    pass

# from tlkit.utils import TASKS_TO_CHANNELS, FEED_FORWARD_TASKS

MAKE_RESCALE_0_1_NEG1_POS1   = lambda n_chan: transforms.Normalize([0.5]*n_chan, [0.5]*n_chan)
RESCALE_0_1_NEG1_POS1        = transforms.Normalize([0.5], [0.5])  # This needs to be different depending on num out chans
MAKE_RESCALE_0_MAX_NEG1_POS1 = lambda maxx: transforms.Normalize([maxx / 2.], [maxx * 1.0])
RESCALE_0_255_NEG1_POS1      = transforms.Normalize([127.5,127.5,127.5], [255, 255, 255])
MAKE_RESCALE_0_MAX_0_POS1 = lambda maxx: transforms.Normalize([0.0], [maxx * 1.0])


def sobel_transform(x):
    x = transforms.functional.to_tensor(x)
    image = x.numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    sx = ndimage.sobel(blur, axis=0, mode='constant')
    sy = ndimage.sobel(blur, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    edge = torch.FloatTensor(sob).unsqueeze(0)
    return edge


def get_transform(task: str, image_size=Optional[int]):
   
    if task in ['rgb', 'normal', 'reshading']:
        transform = transform_8bit
    elif task in ['mask_valid']:
        transform = transforms.ToTensor()
    elif task in ['keypoints2d', 'keypoints3d', 'depth_euclidean', 'depth_zbuffer', 'edge_texture', 'edge_occlusion']:
#         return transform_16bit_int
        transform = transform_16bit_single_channel
    elif task in ['principal_curvature', 'curvature']:
        transform = transform_8bit_n_channel(2)
    elif task in ['segment_semantic']:  # this is stored as 1 channel image (H,W) where each pixel value is a different class
        transform = transform_dense_labels
#     elif len([t for t in FEED_FORWARD_TASKS if t in task]) > 0:
#         return torch.Tensor
#     elif 'decoding' in task:
#         return transform_16bit_n_channel(TASKS_TO_CHANNELS[task.replace('_decoding', '')])
#     elif 'encoding' in task:
#         return torch.Tensor
    elif task in ['class_object', 'class_scene']:
        transform = torch.Tensor
        image_size = None
    elif task == '2d_edges':
        transform = sobel_transform
    else:
        raise NotImplementedError("Unknown transform for task {}".format(task))
    
    if 'clamp_to' in task_configs.task_parameters[task]:
        minn, maxx = task_configs.task_parameters[task]['clamp_to']
        if minn > 0:
            raise NotImplementedError("Rescaling (min1, max1) -> (min2, max2) not implemented for min1, min2 != 0 (task {})".format(task))
        transform = transforms.Compose([
                        transform,
                        MAKE_RESCALE_0_MAX_0_POS1(maxx)])

    if image_size is not None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transform])

    return transform

# For semantic segmentation
transform_dense_labels = lambda img: torch.Tensor(np.array(img)).long()  # avoids normalizing

# Transforms to a 3-channel tensor and then changes [0,1] -> [-1, 1]
transform_8bit = transforms.Compose([
        transforms.ToTensor(),
#         MAKE_RESCALE_0_1_NEG1_POS1(3),
    ])
    
# Transforms to a n-channel tensor and then changes [0,1] -> [-1, 1]. Keeps only the first n-channels
def transform_8bit_n_channel(n_channel=1, crop_channels=False):
    if crop_channels:
        crop_channels_fn = lambda x: x[:n_channel] if x.shape[0] > n_channel else x
    else: 
        crop_channels_fn = lambda x: x
    return transforms.Compose([
            transforms.ToTensor(),
            crop_channels_fn,
#             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])

# Transforms to a 1-channel tensor and then changes [0,1] -> [-1, 1].
def transform_16bit_single_channel(im):
    im = transforms.ToTensor()(im)
    im = im.float() / (2 ** 16 - 1.0) 
#     return RESCALE_0_1_NEG1_POS1(im)
    return im


def transform_16bit_n_channel(n_channel=1):
    if n_channel == 1:
        return transform_16bit_single_channel # PyTorch handles these differently
    else:
        return transforms.Compose([
            transforms.ToTensor(),
#             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])

# from torchvision import get_image_backend, set_image_backend
# import accimage
# set_image_backend('accimage')
import torchvision.io


def convertrgb2lab(img):
    # convert RGB PIL image to Lab, return the L channel, range: 0-255
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(img, rgb2lab)
    L, a, b = Lab.split()
    return L


def default_loader(path):
    if '.npy' in path:
        return np.load(path)
    elif '.json' in path:
        raise NotImplementedError("Not sure how to load files of type: {}".format(os.path.basename(path)))
    else:
        return pil_loader(path)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(img.mode)
        # return img.convert(img.mode)
        # return img.convert('RGB')

# Faster than pil_loader, if accimage is available
def accimage_loader(path):
    return  accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


