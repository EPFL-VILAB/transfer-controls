import torch.nn.functional as F
import torch

def make_valid_mask(mask_float, image_size, max_pool_size=4):
    '''
        Creates a mask indicating the valid parts of the image(s).
        Enlargens masked area using a max pooling operation.

        Args:
            mask_float: A (b x c x h x w) mask as loaded from the Taskonomy loader.
            max_pool_size: Parameter to choose how much to enlarge masked area.
    '''
    _, _, h, w = mask_float.shape
    mask_float = 1 - mask_float
    mask_float = F.max_pool2d(mask_float, kernel_size=max_pool_size)
    mask_float = F.interpolate(mask_float, (h, w), mode='nearest')
    mask_valid = mask_float == 0
    return mask_valid

def ensure_valid_mask(batch, image_size, if_missing_return_ones_like=None):
        if 'mask_valid' in batch:
            return make_valid_mask(batch['mask_valid'], image_size)
        elif if_missing_return_ones_like is not None:
            shape = list(if_missing_return_ones_like.shape)
            shape[1] = 1
            return torch.ones(shape,
                            dtype=torch.bool,
                            device=if_missing_return_ones_like.device)