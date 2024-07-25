"""Classification head module."""
import torch.nn as nn
import torch 

class ClassificationHead(nn.Module):
    """
    Classification head for converting backbone features to class probabilities.

    Args:
        num_features: Number of input features from the backbone.
        num_classes: Number of classes for classification.
        fc1_out_features: Number of output features of the first fully connected layer.
    """

    def __init__(self, num_features: int, num_classes: int, fc1_out_features: int):
        """Initialization rutene. """

        super().__init__()
        self.fc1 = nn.Linear(num_features, fc1_out_features)
        self.bn1 = nn.BatchNorm1d(fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification head.

        Args:
            input_tensor: Input tensor.

        Returns:
            Output tensor with class probabilities.
        """
        out_tensor = nn.functional.relu(self.bn1(self.fc1(input_tensor)))
        output = self.fc2(out_tensor)
        return output

