import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset

def download_cifar10(data_folder: str = "../data/cifar10") -> Dataset:
    """Downloads CIFAR-10 dataset or uses local files if available.

    Args:
        data_folder (str): Path where the dataset will be saved or loaded from.

    Returns:
        Dataset: A torchvision dataset object for CIFAR-10.
    """
    
    os.makedirs(data_folder, exist_ok=True)

    # Transform to resize CIFAR-10 images to 224x224 (for ViT compatibility)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.ToTensor(),
    ])

    # Check if dataset already exists
    if os.path.exists(os.path.join(data_folder, 'cifar-10-batches-py')):
        print(f"Dataset already exists in {data_folder}. Skipping download.")
    else:
        try:
            CIFAR10(root=data_folder, train=False, transform=transform, download=True)
            print(f"Downloaded CIFAR-10 dataset to {data_folder}")
        except Exception as e:
            print(f"Failed to download the dataset. Error: {e}")
            raise

    # Return the dataset object
    return CIFAR10(root=data_folder, train=False, transform=transform, download=False)



def visualize_image_grid(dataset: Dataset, num_images: int=25):
    """Visualize a grid of images from the dataset.

    Args:
        dataset: The dataset to visualize (e.g., CIFAR-10).
        num_images (int): Number of images to display in the grid.
    """

    # Select a subset of images from the dataset
    images = [dataset[i][0] for i in range(num_images)]  # Get images only (not labels)

    # Create a grid of images
    grid = make_grid(images, nrow=int(num_images**0.5), padding=2)
    
    # Plot the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))  # Convert from CHW to HWC for plotting
    plt.axis("off")
    plt.title(f"Sample of {num_images} Images from Dataset")
    plt.show()

# Download or use the local dataset
cifar10_dataset = download_cifar10()
visualize_image_grid(dataset=cifar10_dataset, num_images=25)