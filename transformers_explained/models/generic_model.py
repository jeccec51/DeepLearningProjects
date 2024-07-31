"""Generic Classification Module."""
import torch
import torch.nn as nn
from omegaconf import DictConfig
from layers.classification_head import ClassificationHead
from models.cnn import CNNBackbone
from models.resnet import  get_resnet_back_bone
from models.vit import VisionTransformerBackbone

class GenericClassifier(nn.Module):
    """Generic classifier that combines a backbone with a classification head.
    
    Args:
        backbone: Backbone network
        num_features: Number of features
        num_classes: Number of classes
        fc1_out_features: Number of out channels from first fully connected layer
    """

    def __init__(self, backbone: nn.Module, num_features: int, num_classes: int,  fc1_out_features: int):
        """Initialization routene. """

        super().__init__()
        self.backbone = backbone
        self.classifier = ClassificationHead(num_features=num_features, num_classes=num_classes, fc1_out_features=fc1_out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Args:
            input_tensor: Input tensor.

        Returns:
            Output tensor with class probabilities.
        """

        features = self.backbone(input_tensor)
        if isinstance(self.backbone, CNNBackbone):
            features = features.view(features.size(0), -1)
        elif isinstance(self.backbone, VisionTransformerBackbone):
            pass # VIT is expected to return 2D tensor 
        else:  # ResNet or ViT
            features = features.mean(dim=[2, 3])
        output = self.classifier(features)
        return output

def get_model(config: DictConfig) -> nn.Module:
    """Get the model based on the configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        The model.
    """
    
    model_type = config.model.name
    fc1_out_features = config.layers.classification_head.fc1_out_features
    if model_type == 'cnn':
        backbone = CNNBackbone(
            conv1_out_channels=config.model.conv1_out_channels,
            conv2_out_channels=config.model.conv2_out_channels
        )
        num_features = config.model.conv2_out_channels * (config.model.img_size // 4) ** 2
        num_classes = config.model.num_classes
    elif model_type == 'resnet':
        backbone = get_resnet_back_bone()
        num_features = backbone[-1][-1].bn2.num_features
        num_classes = config.model.num_classes
    elif model_type == 'vit':
        backbone = VisionTransformerBackbone(
            image_size=config.model.img_size,
            patch_size=config.model.patch_size,
            emb_size=config.model.emb_size,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            dropout_rate=config.model.drop_out_rate
        )
        num_features = config.model.emb_size
        num_classes = config.model.num_classes
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return GenericClassifier(backbone=backbone, num_features=num_features, 
                             num_classes=num_classes, fc1_out_features=fc1_out_features)
