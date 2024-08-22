"""Bifpn Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class BiFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        """Bidirectional Feature Pyramid Network (BiFPN) for multi-scale feature fusion.

        Args:
            in_channels: Number of input channels for the initial feature maps.
            out_channels: Number of output channels for the fused feature maps.
            num_layers: Number of layers to stack in BiFPN.
        """

        super().__init__()

        # Dynamic creation of convolution layers based on configuration
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # Learnable weights for adaptive feature fusion
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) for _ in range(num_layers)
        ])

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through the BiFPN.

        Args:
            feature_maps: List of feature maps from EfficientNet backbone.

        Returns:
             List of fused feature maps.
        """
        
        # Assuming feature_maps length is equal to num_layers
        outputs = []
        for i, feature_map in enumerate(feature_maps):
            weight = F.relu(self.weights[i])
            weight = weight / (torch.sum(weight) + 0.0001)
            output = self.convs[i](feature_map)
            outputs.append(output * weight[0] + F.interpolate(output, scale_factor=2) * weight[1])

        return outputs

if __name__ == "__main__":
    # Example usage of BiFPN
    feature_maps = [
        torch.randn(1, 64, 56, 56),
        torch.randn(1, 128, 28, 28),
        torch.randn(1, 256, 14, 14),
        torch.randn(1, 512, 7, 7),
        torch.randn(1, 1024, 4, 4)
    ]
    bifpn = BiFPN(in_channels=64, out_channels=128, num_layers=5)
    output_maps = bifpn(feature_maps)
    for i, feature_map in enumerate(output_maps):
        print(f"Output feature map {i} shape: {feature_map.shape}")
