import torch
import torch.nn as nn

from . import layers


class ResNet(nn.Module):
  """Residual neural network."""

  def __init__(self,  # pylint: disable=too-many-arguments
               num_in_channels: int,
               num_out_channels: int,
               num_features: int = 64,
               num_blocks: int = 9,
               num_convs_per_block: int = 2) -> None:
    super(ResNet, self).__init__()

    resnet_list = [layers.Conv(num_in_channels, num_features)]
    for _ in range(num_blocks):
      resnet_list.append(layers.ResBlock(
          num_features, num_features, num_convs=num_convs_per_block))
    resnet_list.append(layers.Conv(num_features, num_out_channels))

    self.resnet = nn.Sequential(*resnet_list)

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    return self.resnet(features)