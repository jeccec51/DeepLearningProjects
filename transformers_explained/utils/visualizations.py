"""Visu Module."""

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
        _ = model.backbone(images)
        attentions = model.backbone.get_attention_map()

    # Plot attention maps
    fig, axes = plt.subplots(len(images), len(attentions[0]), figsize=(15, 10))
    for i, img in enumerate(images):
        for j, attention in enumerate(attentions[0]):
            axes[i, j].imshow(img.cpu().permute(1, 2, 0))  # Original image
            axes[i, j].imshow(np.mean(attention[i], axis=0), cmap='viridis', alpha=0.6)  # Attention map overlay
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'Head {j+1}')

    plt.suptitle('Attention Maps Visualization')
    plt.show()
