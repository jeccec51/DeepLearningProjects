"""Generic Classification Module."""
import torch
from torch import nn
from layers.classification_head  import ClassificationHead
from models.cnn import CNNBackbone
from models.resnet import  get_resnet_back_bone
from models.vit import  VisionTransformerBackbone
from omegaconf import DictConfig


class GenericClassifier(nn.Module):
    """Generic classifier that combines a backbone with a classification head.

    Args:
        backbone: Feature extraction backbone.
        num_classes: Number of classes for classification.
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        """initialization routene."""

        super().__init__()
        self.backbone = backbone
        if isinstance(backbone, CNNBackbone):
            num_features = 32 * 56 * 56
        elif isinstance(backbone, VisionTransformerBackbone):
            num_features = backbone.patch_embedding.proj.out_channels
        else:  # ResNet or similar
            num_features = backbone[-1][-1].bn2.num_features
        self.classifier = ClassificationHead(num_features, num_classes)
        

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Args:
            in_tensor: Input tensor.

        Returns:
            Output tensor with class probabilities.
        """

        out_tensor = self.backbone(in_tensor)
        if isinstance(self.backbone, CNNBackbone):
            out_tensor = out_tensor.view(out_tensor.size(0), -1)
        else:  # ResNet or ViT
            out_tensor = out_tensor.mean(dim=[2, 3])
        out_tensor = self.classifier(out_tensor)
        return out_tensor

def get_model(config: DictConfig) -> nn.Module:
    """Get the model based on the configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        The model.
    """

    model_type = config.model.type
    if model_type == 'cnn':
        backbone = CNNBackbone()
    elif model_type == 'resnet':
        backbone = get_resnet_back_bone()
    elif model_type == 'vit':
        backbone = VisionTransformerBackbone(img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            emb_size=config.model.emb_size,
            depth=config.model.depth,
            num_heads=config.model.num_heads
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return GenericClassifier(backbone, config.model.num_classes)
