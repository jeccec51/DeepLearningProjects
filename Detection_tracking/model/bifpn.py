"""Module to implement bifpn network architecture. """
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Bidirectional Feature Pyramid Network (BiFPN) for multi-scale feature fusion.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """

        super(BiFPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through the BiFPN.

        Args:
            feature_maps: List of feature maps from EfficientNet backbone.

        Returns:
            List of fused feature maps.
        """
        
        p3, p4, p5, p6, p7 = feature_maps
        
        # Fusion of features at different levels
        p6_out = self.conv1(p6) + self.conv2(F.interpolate(p7, scale_factor=2))
        p5_out = self.conv3(p5) + self.conv4(F.interpolate(p6_out, scale_factor=2))
        p4_out = self.conv5(p4) + F.interpolate(p5_out, scale_factor=2)
        
        return [p3, p4_out, p5_out, p6_out, p7]

if __name__ == "__main__":
    bifpn = BiFPN(in_channels=40, out_channels=64)
    print(bifpn)
