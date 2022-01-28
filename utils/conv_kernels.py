import numpy as np


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # From https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_sobel_kernel(k=3):
    # From https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_laplace_kernel(diag_filter=False):
    if diag_filter:
        return np.array([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]])
    else:
        return np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])

def get_emboss_kernel():
    return np.array([[0.,0.,0.],[0.,1.0,0.],[-1.,0.,0.]])

def get_4d_emboss_kernels():
    return np.array([
        [[0,1,0], [0,0,0], [0,-1,0]],
        [[1,0,0], [0,0,0], [0,0,-1]],
        [[0,0,0], [1,0,-1], [0,0,0]],
        [[0,0,1], [0,0,0], [-1,0,0]]
    ])