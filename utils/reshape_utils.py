import numpy as np
import torch


def cubify(arr, newshape):
    '''
    Breaks up an n-dimensional array of shape (D_1, D_2, ..., D_n) into N blocks of shape (d_1, d_2, ..., d_n).
    Each block dimension d_i must divide the corresponding array dimension D_i without remainder.

    See 'utils.decubify' for the inverse function.

    From https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440

    Args:
        arr: A Numpy array or PyTorch Tensor of shape (D_1, D_2, ..., D_n).
        newshape: A tuple (d_1, d_2, ..., d_n) describing the shape of each of the N blocks.
    Returns:
        A new array of shape (N, d_1, d_2, ..., d_n).
    '''
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    if isinstance(arr, torch.Tensor):
        return arr.reshape(tuple(tmpshape)).permute(tuple(order)).reshape(-1, *newshape)
    else:
        return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def decubify(arr, oldshape):
    '''
    Reassembles N blocks of shape (d_1, d_2, ..., d_n) into their original array of shape (D_1, D_2, ..., D_n).

    See 'utils.cubify' for the inverse function.

    From https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440

    Args:
        arr: A Numpy array or PyTorch Tensor of shape (N, d_1, d_2, ..., d_n).
        oldshape: A tuple (D_1, D_2, ..., D_n) describing the shape of the original array.
    Returns:
        The reconstructed array of shape (D_1, D_2, ..., D_n).
    '''
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    if isinstance(arr, torch.Tensor):
        return arr.reshape(tuple(tmpshape)).permute(tuple(order)).reshape(tuple(oldshape))
    else:
        return arr.reshape(tmpshape).transpose(order).reshape(oldshape)