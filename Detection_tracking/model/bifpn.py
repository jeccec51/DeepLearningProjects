"""Bifpn Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Bidirectional Feature Pyramid Network (BiFPN) for multi-scale feature fusion.

        Args:
            in_channels: Number of input channels for the initial feature maps.
            out_channels: Number of output channels for the fused feature maps.
        """

        super(BiFPN, self).__init__()

        # Convolution layers for fusing features at different levels
        self.conv3_to_4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4_to_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv5_to_6 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv6_to_7 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4_to_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv5_to_4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv6_to_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv7_to_6 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Learnable weights for adaptive feature fusion
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w6 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w7 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


    def forward(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through the BiFPN.

        Args:
            feature_maps: List of feature maps from EfficientNet backbone.

        Returns:
             List of fused feature maps.
        """

        feature_map_3, feature_map_4, feature_map_5, feature_map_6, feature_map_7 = feature_maps

        # Normalize the weights for the top-down pathway
        weight_3 = F.relu(self.w3)
        weight_3 = weight_3 / (torch.sum(weight_3) + 0.0001)
        weight_4 = F.relu(self.w4)
        weight_4 = weight_4 / (torch.sum(weight_4) + 0.0001)
        weight_5 = F.relu(self.w5)
        weight_5 = weight_5 / (torch.sum(weight_5) + 0.0001)
        weight_6 = F.relu(self.w6)
        weight_6 = weight_6 / (torch.sum(weight_6) + 0.0001)
        weight_7 = F.relu(self.w7)
        weight_7 = weight_7 / (torch.sum(weight_7) + 0.0001)

        # Top-down pathway
        feature_map_6_up = self.conv6_to_5(feature_map_6) + self.conv7_to_6(F.interpolate(feature_map_7, scale_factor=2))
        feature_map_5_up = self.conv5_to_4(feature_map_5) + self.conv6_to_5(F.interpolate(feature_map_6_up, scale_factor=2))
        feature_map_4_up = self.conv4_to_3(feature_map_4) + self.conv5_to_4(F.interpolate(feature_map_5_up, scale_factor=2))
        feature_map_3_up = self.conv3_to_4(feature_map_3) + F.interpolate(feature_map_4_up, scale_factor=2)

        # Bottom-up pathway
        feature_map_4_out = self.conv4_to_3(feature_map_4_up) + \
                            weight_4[0] * self.conv4_to_3(feature_map_4) + \
                            weight_4[1] * F.interpolate(feature_map_5_up, scale_factor=0.5) + \
                            weight_4[2] * F.interpolate(feature_map_3_up, scale_factor=0.5)
        
        feature_map_5_out = self.conv5_to_4(feature_map_5_up) + \
                            weight_5[0] * self.conv5_to_4(feature_map_5) + \
                            weight_5[1] * F.interpolate(feature_map_6_up, scale_factor=0.5) + \
                            weight_5[2] * F.interpolate(feature_map_4_out, scale_factor=2)
        
        feature_map_6_out = self.conv6_to_5(feature_map_6_up) + \
                            weight_6[0] * self.conv6_to_5(feature_map_6) + \
                            weight_6[1] * F.interpolate(feature_map_5_out, scale_factor=2)

        return [feature_map_3_up, feature_map_4_out, feature_map_5_out, feature_map_6_out, feature_map_7]

if __name__ == "__main__":
    # Example usage of BiFPN
    feature_maps = [
        torch.randn(1, 64, 56, 56),
        torch.randn(1, 128, 28, 28),
       ```python
        torch.randn

