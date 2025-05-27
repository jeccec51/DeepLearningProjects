from cProfile import label
from email.mime import image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import matplotlib.pyplot as plt

from transformers_explained.utils import data_loader

# Load image and apply transform

def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extracts patches from an image tensor. 
    
    Args:
        image (torch.Tensor): The input image tensor of shape (C, H, W).
        patch_size (int): The size of each patch.
    Returns:
        torch.Tensor: A tensor of patches of shape (num_patches, C, patch_size, patch_size).
    """

    _, height, width = image.shape

    assert height % patch_size == 0 and width % patch_size == 0, \
        "image dimentions must be divisible by patch size"
    
    # Extract patches using unfold
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, image.shape[0], patch_size, patch_size)

    return patches

def load_data(data_folder: str="../data/cifar10")->DataLoader:
    """Loads CIFAR-10 datasets from a dataloader.
    Args:
        data_folder (str): Path to CIFAR-10 datasets.
    Returns:
        DataLoader: A DataLoader object containing the CIFAR-10 dataset.
    """

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                    ])
    dataset = CIFAR10(root=data_folder, train=False, transform=transform, download=False)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    return data_loader

def visualize_patches(patches: torch.Tensor, num_patches:int = 10)->None:
    """Visualizes a few patches from the extracted patches.

    Args:
        patches (torch.Tensor): The tensor containing image patches.
        num_patches (int, optional): Number of patches to visualize. Defaults to 10.
    """

    # Ensure num_patches does not exceed the total number of patches
    num_patches = min(num_patches, patches.shape[0])

    fig, axes = plt.subplots(1, num_patches, figsize=(num_patches * 2, 2))
    for i in range(num_patches):
        patch = patches[i].permute(1, 2, 0)  # Change shape to (H, W, C) for visualization
        axes[i].imshow(patch)
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    #load data set
    data_folder = "../data/cifar10"
    data_loader = load_data(data_folder=data_folder)

    # Fetch one image from the data loader
    for bath in data_loader:
        image, label = bath
        image = image.squeeze(0)
        print(f"Loaded image Shape:{image.shape}, ")