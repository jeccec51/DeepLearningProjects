"""Cnn based feature extraction back bone."""

import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNBackbone(nn.Module):
    """CNN backbone for feature extraction.
    
    Args:
        conv1_out_channels: output channels of first convolution layer
        conv2_out_channels: output channels of secind conv layer
    """

    def __init__(self, conv1_out_channels: int, conv2_out_channels: int):
        """Initialization routene. """

        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN backbone.

        Args:
            input_tensor: Input tensor.

        Returns:
            Output tensor after passing through the network.
        """

        pooled_output_1 = self.pool(F.relu(self.bn1(self.conv1(input_tensor))))
        pooled_output_2 = self.pool(F.relu(self.bn2(self.conv2(pooled_output_1))))
        return pooled_output_2    