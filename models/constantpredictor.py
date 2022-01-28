import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConstantPredictor(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.estimate = torch.nn.parameter.Parameter(torch.zeros(shape, dtype=torch.float32))
        
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.stack([self.estimate] * batch_size, dim=0)

    