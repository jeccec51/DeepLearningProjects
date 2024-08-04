"""Efficientne backbone module."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class EfficientNetBackbone(nn.Module):
    """EfficientNet Backbone for feature extraction.
       Note:
            Utilizes EfficientNet-B0 from torchvision. """
    

    def __init__(self):
        """Initialization Routene."""

        super(EfficientNetBackbone, self).__init__()

        # Load the pretrained EfficientNet-B0 model
        efficientnet_model = efficientnet_b0(pretrained=True)

        # Extract the feature layers from EfficientNet-B0
        self.initial_conv = nn.Sequential(*efficientnet_model.features[0:2])  # Initial Convolution and BatchNorm
        self.mbconv1 = efficientnet_model.features[2]                        # MBConv1
        self.mbconv6_1 = efficientnet_model.features[3]                      # MBConv6 (first block)
        self.mbconv6_2 = efficientnet_model.features[4]                      # MBConv6 (second block)
        self.mbconv6_3 = efficientnet_model.features[5]                      # MBConv6 (third block)
        self.mbconv6_4 = efficientnet_model.features[6]                      # MBConv6 (fourth block)
        self.mbconv6_5 = efficientnet_model.features[7]                      # MBConv6 (fifth block)


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the EfficientNet backbone.
        
        Args:
            input_tensor: Input tensor of shape (N, C, H, W)
        
        Returns:
            Output tensor after passing through EfficientNet features
        """

        # Initial Convolution and BatchNorm
        features = self.initial_conv(input_tensor)
        print(f"Initial Convolution output shape: {features.shape}")

        # MBConv1
        features = self.mbconv1(features)
        print(f"MBConv1 output shape: {features.shape}")

        # MBConv6 (first block)
        features = self.mbconv6_1(features)
        print(f"MBConv6 (first block) output shape: {features.shape}")

        # MBConv6 (second block)
        features = self.mbconv6_2(features)
        print(f"MBConv6 (second block) output shape: {features.shape}")

        # MBConv6 (third block)
        features = self.mbconv6_3(features)
        print(f"MBConv6 (third block) output shape: {features.shape}")

        # MBConv6 (fourth block)
        features = self.mbconv6_4(features)
        print(f"MBConv6 (fourth block) output shape: {features.shape}")

        # MBConv6 (fifth block)
        features = self.mbconv6_5(features)
        print(f"MBConv6 (fifth block) output shape: {features.shape}")
        
        return features


if __name__ == "__main__":
    # Create an instance of EfficientNetBackbone and print its architecture
    model = EfficientNetBackbone()
    print(model)

    # Test the model with a random input tensor
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size
    output_tensor = model(input_tensor)
    print(f"Output tensor shape: {output_tensor.shape}")
