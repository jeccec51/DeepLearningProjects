"""Bifpn Module."""

import torch
from torch import nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Bidirectional Feature Pyramid Network (BiFPN) for multi-scale feature fusion.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """

        super().__init__()

        # Convolution layers for feature fusion
        self.conv6_to_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv5_to_4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4_to_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_to_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2_to_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Learnable weights for weighted feature fusion
        self.weight_top_down_6 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weight_top_down_5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weight_top_down_4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weight_bottom_up_3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.weight_bottom_up_4 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.weight_bottom_up_5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


    def forward(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through the BiFPN.

        Args:
            feature_maps: List of feature maps from EfficientNet backbone.

        Returns:
            List of fused feature maps.
        """
        feature_map_3, feature_map_4, feature_map_5, feature_map_6, feature_map_7 = feature_maps

        # Normalize weights for top-down pathway
        weight_top_down_6 = F.relu(self.weight_top_down_6)
        weight_top_down_6 = weight_top_down_6 / (torch.sum(weight_top_down_6) + 0.0001)
        weight_top_down_5 = F.relu(self.weight_top_down_5)
        weight_top_down_5 = weight_top_down_5 / (torch.sum(weight_top_down_5) + 0.0001)
        weight_top_down_4 = F.relu(self.weight_top_down_4)
        weight_top_down_4 = weight_top_down_4 / (torch.sum(weight_top_down_4) + 0.0001)

        # Top-down pathway
        feature_map_6_up = self.conv6_to_5(feature_map_6) + self.conv5_to_4(F.interpolate(feature_map_7, scale_factor=2))
        feature_map_5_up = self.conv5_to_4(feature_map_5) + self.conv4_to_3(F.interpolate(feature_map_6_up, scale_factor=2))
        feature_map_4_up = self.conv4_to_3(feature_map_4) + self.conv3_to_2(F.interpolate(feature_map_5_up, scale_factor=2))
        feature_map_3_up = self.conv3_to_2(feature_map_3) + F.interpolate(feature_map_4_up, scale_factor=2)

        # Normalize weights for bottom-up pathway
        weight_bottom_up_3 = F.relu(self.weight_bottom_up_3)
        weight_bottom_up_3 = weight_bottom_up_3 / (torch.sum(weight_bottom_up_3) + 0.0001)
        weight_bottom_up_4 = F.relu(self.weight_bottom_up_4)
        weight_bottom_up_4 = weight_bottom_up_4 / (torch.sum(weight_bottom_up_4) + 0.0001)
        weight_bottom_up_5 = F.relu(self.weight_bottom_up_5)
        weight_bottom_up_5 = weight_bottom_up_5 / (torch.sum(weight_bottom_up_5) + 0.0001)

        # Bottom-up pathway
        feature_map_4_out = self.conv4_to_3(feature_map_4_up) + \
                            weight_bottom_up_3[0] * self.conv4_to_3(feature_map_4) + \
                            weight_bottom_up_3[1] * F.interpolate(feature_map_5_up, scale_factor=0.5) + \
                            weight_bottom_up_3[2] * F.interpolate(feature_map_3_up, scale_factor=0.5)
        
        feature_map_5_out = self.conv5_to_4(feature_map_5_up) + \
                            weight_bottom_up_4[0] * self.conv5_to_4(feature_map_5) + \
                            weight_bottom_up_4[1] * F.interpolate(feature_map_6_up, scale_factor=0.5) + \
                            weight_bottom_up_4[2] * F.interpolate(feature_map_4_out, scale_factor=2)
        
        feature_map_6_out = self.conv6_to_5(feature_map_6_up) + \
                            weight_bottom_up_5[0] * self.conv6_to_5(feature_map_6) + \
                            weight_bottom_up_5[1] * F.interpolate(feature_map_5_out, scale_factor=2)

        return [feature_map_3_up, feature_map_4_out, feature_map_5_out, feature_map_6_out, feature_map_7]
        

if __name__ == "__main__":
    # Example usage of BiFPN
    feature_maps = [
        torch.randn(1, 64, 56, 56),
        torch.randn(1, 128, 28, 28),
        torch.randn(1, 256, 14, 14),
        torch.randn(1, 512, 7, 7),
        torch.randn(1, 1024, 4, 4)
    ]
    bifpn = BiFPN(in_channels=64, out_channels=128)
    output_maps = bifpn(feature_maps)
    for i, feature_map in enumerate(output_maps):
        print(f"Output feature map {i} shape: {feature_map.shape}")
