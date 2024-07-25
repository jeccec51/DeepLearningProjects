"""Classification head module."""
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """Classification head for converting backbone features to class probabilities.

    Args:
        num_features: Number of input features from the backbone.
        num_classes: Number of classes for classification.
    """
    def __init__(self, num_features: int, num_classes: int) -> None:
        """Initialization routene. """
        super().__init__()
        self.fc1 = nn.Linear(num_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification head.

        Args:
            in_tensor: Input tensor.

        Returns:
            Output tensor with class probabilities.
        """

        out_tensor = nn.functional.relu(self.bn1(self.fc1(in_tensor)))
        out_tensor = self.fc2(out_tensor)
        return out_tensor
