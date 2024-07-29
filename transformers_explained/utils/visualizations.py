"""Visu Module."""

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_feature_maps(model, loader):
    """Visualize feature maps of a model.

    Args:
        model: The trained model.
        loader: Data loader to fetch data from.
    """

    model.eval()
    dataiter = iter(loader)
    images, _ = next(dataiter)  # Fetch a batch of images
    images = images[:4]  # Take the first 4 images for visualization

    # Send images to the same device as the model
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        outputs = model.backbone(images)

    # Plot feature maps
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 10))
    for i, img in enumerate(images):
        axes[i, 0].imshow(img.cpu().permute(1, 2, 0))  # Original image
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Original Image {i+1}')

        feature_map = outputs[i].cpu().numpy()
        feature_map = np.mean(feature_map, axis=0)  # Average over all channels
        axes[i, 1].imshow(feature_map, cmap='viridis')  # Feature map
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Feature Map {i+1}')

    plt.suptitle('Feature Maps Visualization')
    plt.show()


def visualize_attention_maps(model, loader):
    """Visualize attention maps of a Vision Transformer model.

    Args:
        model: The trained Vision Transformer model.
        loader: Data loader to fetch data from.
    """
    
    model.eval()
    dataiter = iter(loader)
    images, _ = next(dataiter)  # Fetch a batch of images
    images = images[:4]  # Take the first 4 images for visualization

    # Send images to the same device as the model
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        attentions = model.backbone.get_attention_map(images)

    # Plot attention maps
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 10))
    for i, img in enumerate(images):
        axes[i, 0].imshow(img.cpu().permute(1, 2, 0))  # Original image
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Original Image {i+1}')

        attention_map = attentions[i].cpu().numpy()
        attention_map = np.mean(attention_map, axis=0)  # Average over all heads
        axes[i, 1].imshow(attention_map, cmap='viridis')  # Attention map
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Attention Map {i+1}')

    plt.suptitle('Attention Maps Visualization')
    plt.show()
