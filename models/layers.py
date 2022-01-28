"""Neural network layers and components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
  """Conv layer followed by leaky ReLU."""

  def __init__(self, num_in_channels: int, num_out_channels: int) -> None:
    super(Conv, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(num_in_channels, num_out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
    )

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    return self.conv(features)


class ConvBlock(nn.Module):
  """Conv block containing multiple conv layers."""

  def __init__(self, num_in_channels: int, num_out_channels: int,
               num_convs: int = 2) -> None:
    super(ConvBlock, self).__init__()

    conv_block_list = [Conv(num_in_channels, num_out_channels)]
    for _ in range(num_convs - 1):
      conv_block_list.append(Conv(num_out_channels, num_out_channels))
    self.conv_block = nn.Sequential(*conv_block_list)

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    return self.conv_block(features)


class ResBlock(nn.Module):
  """Residual block containing conv block with a residual connection."""

  def __init__(self, num_in_channels: int, num_out_channels: int,
               num_convs: int = 2) -> None:
    super(ResBlock, self).__init__()

    self.conv_block = ConvBlock(
        num_in_channels, num_out_channels, num_convs=num_convs)

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    return features + self.conv_block(features)


class DownsampleBlock(nn.Module):
  """Downsample block containing multiple convs followed by a downsample."""

  def __init__(self, num_in_channels: int, num_out_channels: int,
               num_convs: int = 2) -> None:
    super(DownsampleBlock, self).__init__()
    self.conv_block = ConvBlock(num_in_channels, num_out_channels, num_convs)

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    features = self.conv_block(features)
    features = F.interpolate(features, scale_factor=0.5, mode='area')
    return features


class UpsampleBlock(nn.Module):
  """Upsample block containing an upsample followed by multiple convs."""

  def __init__(self, num_in_channels: int, num_out_channels: int,
               num_convs: int = 3) -> None:
    super(UpsampleBlock, self).__init__()
    self.conv_block = ConvBlock(num_in_channels, num_out_channels, num_convs)

  def forward(  # pylint: disable=arguments-differ
      self, features: torch.Tensor) -> torch.Tensor:
    features = F.interpolate(
        features, scale_factor=2.0, mode='bilinear', align_corners=False)
    features = self.conv_block(features)
    return features