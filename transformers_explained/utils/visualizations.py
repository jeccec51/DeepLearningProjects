"""Visu Module."""

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

def visualize_feature_maps(model: nn.Module, data_loader: DataLoader) -> None:
    """Visualize feature maps from the first two convolutional layes of CNN.
    
    Args:
        model: The CNN Model
        data_loader: Data loader to get input images from

    """

    def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Hook Function to capture output of a layer.
        
        Args:
            module: the module being hooked
            input: input to the module
            output: output of the ,module
        """
        feature_maps.append(output)

    feature_maps = []  
    model.backbone.conv1.register_forward_hook(hook)
    model.backbone.conv2.register_forward_hook(hook)

    dataiter = iter(data_loader)
    images, _ = dataiter.next()
    _ = model(images)

    for fmap in feature_maps:
        fmap = fmap.detach().cpu().numpy()
        fig, axs = plt.subplots(1, fmap.shape[1], figsize=(20, 5))
        for index in range(fmap.shape[1]):
            axs[index].imshow(fmap[0, index], cmap='gray')
            axs[index].axis('off')
        plt.show()


def visualize_attention_maps(model: nn.Module, dataloader: DataLoader):
    """Visualize attention maps from the Vision Transformer.

    Args:
        model: The Vision Transformer model.
        dataloader: The data loader to get input images from.
    """

    def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook function to capture the output of a layer.

        Args:
            module (nn.Module): The module being hooked.
            input (torch.Tensor): The input to the module.
            output (torch.Tensor): The output of the module.
        """
        attention_maps.append(output)

    attention_maps = []
    model.backbone.transformer_encoder.layers[0].self_attn.register_forward_hook(hook)

    dataiter = iter(dataloader)
    images, _ = dataiter.next()
    _ = model(images)

    for amap in attention_maps:
        amap = amap.detach().cpu().numpy()
        fig, axs = plt.subplots(1, amap.shape[1], figsize=(20, 5))
        for index in range(amap.shape[1]):
            axs[index].imshow(amap[0, index], cmap='hot')
            axs[index].axis('off')
        plt.show()

