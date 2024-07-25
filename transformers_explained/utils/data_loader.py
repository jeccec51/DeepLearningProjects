"""Data loader Module."""

from typing import Tuple
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_data_loaders(dataset_name: str, img_size: int, batch_size: int, use_yuv: bool) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR10").
        img_size: Size to resize the images to.
        batch_size: Batch size for the data loaders.
        use_yuv: Whether to use YUV color space instead of RGB.

    Returns:
        Train and test data loaders.
    """

    if use_yuv:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda x: x.convert('YCbCr')),  # Convert to YUV
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader