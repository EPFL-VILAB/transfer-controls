import math
import torch
import torch.nn.functional as F
import torch_dct as dct

from .reshape_utils import cubify, decubify


def dct_patch(img, zigzag_idxs, n_components):
    '''
    Performs 2D Discrete Cosine Transform on a patch-wise and channel-wise basis over an entire image.

    Args:
        zigzag_idxs: As returned by the function zigzag_indices(patch_size)
        n_components: How many DCT coefficients to return per patch and per channel.
            Must be smaller than patch_size^2.
    Returns:
        DCT coefficients of shape (batch_size x (num_channels*num_components) x num_patches x num_patches)
    '''
    n_batches, n_channels, side_length, side_length = img.shape # B x C x H x W
    patch_size = int(math.sqrt(len(zigzag_idxs[0])))
    n_patches = side_length // patch_size
    img_patches = cubify(img, (n_batches, n_channels, patch_size, patch_size)) # N x B x C x P x P
    coeffs = dct.dct_2d(img_patches) # N x B x C x P x P
    coeffs = zigzagify(coeffs, zigzag_idxs)[...,:n_components] # N x B x C x n_components
    coeffs = coeffs.permute(1,2,3,0).reshape(n_batches, n_channels*n_components, n_patches, n_patches)
    return coeffs

def idct_patch(coeffs, zigzag_idxs, n_components, n_channels, side_length):
    '''
    Performs an inverse patch-wise 2D Discrete Cosine Transform.

    Args:
        coeffs: DCT coefficients of shape (batch_size x (num_channels*num_components) x num_patches x num_patches)
        zigzag_idxs: As returned by the function zigzag_indices(patch_size)
        n_components: How many DCT coefficients were returned by DCT per patch and per channel
        n_channels: Number of color channels in the original image
        side_length: Side length of the original image
    Returns:
        Reconstructed image of shape (batch_size x num_channels x side_length x side_length)
    '''
    n_batches = coeffs.shape[0]
    coeffs = coeffs.reshape(n_batches, n_channels, n_components, -1).permute(3,0,1,2) # N x B x C x n_components
    coeffs = dezigzagify(coeffs, zigzag_idxs) # N x B x C x P x P
    reconst = dct.idct_2d(coeffs) # N x B x C x P x P
    reconst = decubify(reconst, (n_batches, n_channels, side_length, side_length)) # B x C x H x W
    return reconst

def zigzag_indices(n):
    '''
    Computes the zig zag scan indices for an nxn matrix, useful for selecting "most important"
    DCT coefficients.

    Args:
        n: Matrix side length
    Returns:
        List of lists, containing index pairs in sorted zig zag scan order
    '''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    indices = [index for index in sorted(
        ([x, y] for x in xs for y in xs),
        key=compare
    )]
    return torch.LongTensor(indices).transpose(1,0).tolist()

def zigzagify(X, indices):
    '''
    Flattens the last two dimensions of a tensor in a zig zag scan manner.

    Args:
        X: Tensor of arbitrary shape, with last two dimensions having the same dimensions
        indices: Precomputed zig zag scan indices
    Returns:
        Tensor where last two dimensions were flattened into one
    '''
    return X[...,indices[0],indices[1]]

def dezigzagify(v, indices):
    '''
    Undoes the zig zag scan operation and reconstructs a square matrix in the last two dimensions.

    Args:
        v: Tensor or arbitrary shape, where the last dimension are zig zag scan flattened. Last dimension
            may be smaller than the square of the original side length. The remaining coefficients are
            replaced by zeros.
        indices: Precomputed zig zag scan indices
    Returns:
        Tensor where last two dimensions are reconstructed from zig zag flattened vector
    '''
    side_length = int(math.sqrt(len(indices[0])))
    X = torch.zeros(list(v.shape[:-1]) + [side_length, side_length]).to(v.device)
    v = F.pad(v, (0, len(indices[0]) - v.shape[-1]), 'constant', 0)
    X[...,indices[0],indices[1]] = v
    return X

def init_conv_dct(weights):
    '''
    Initialize the weights of a convolutional layer with DCT bases.

    Args:
        weights: Weights of Conv2d module
    '''
    n_filters, n_channels, k, _ = weights.shape
    
    indices = zigzag_indices(k)[:n_filters]
    for idx, (i_x, i_y) in enumerate(zip(indices[0], indices[1])):
        
        c_x = torch.zeros(k)
        c_x[i_x] = 1
        c_y = torch.zeros(k)
        c_y[i_y] = 1
        
        basis = torch.ger(dct.idct(c_x), dct.idct(c_y))
        
        weights[idx,...] = basis