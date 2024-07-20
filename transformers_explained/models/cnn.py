import torch.nn as nn
import torch.nn.functional as F
import torch 

class CNNBackbone(nn.Module):
    """A simple convloutional neural network class for calssification.

    Args:
        num_classes: Numbr of classes

    """
    def __init__(self, num_classes: int) -> None:
        """Initialization Routene."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass of CNN.
        
        Args:
            in_tensor: Input Tensor
        
        Returns: 
            Output tensor after passing through the network."""

        out_tensor = self.pool1(F.relu(self.bn1(self.conv1(in_tensor))))
        out_tensor = self.pool1(F.relu(self.bn2(self.conv2(out_tensor))))
        return out_tensor
    