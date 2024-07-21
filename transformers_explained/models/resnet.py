"""Simple resnet based backbone module."""

from torchvision.models import resnet18
import torch.nn as nn

def get_resnet_back_bone() -> nn.Module:
    """Load resnet-18 backbone for feature extraction.
    
    Returns:
        Resnet-18 Back bone
    """
    model = resnet18(pretrained=True)
    backbone = nn.Sequential(*(list(model.children())[:-2]))
    return backbone
