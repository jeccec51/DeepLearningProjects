"""Data loader Module."""

from typing import Tuple
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_data_loaders(dataset_name: str, img_size: int, batch_size: int, use_yuv: bool, 
                     run_type: str, short_run_fraction: float, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR10").
        img_size: Size to resize the images to.
        batch_size: Batch size for the data loaders.
        use_yuv: Whether to use YUV color space instead of RGB.
        run_type: Type of run, "short" or "long".
        short_run_fraction: Fraction of data to use for a short run.
        num_workers: Number of worker processes to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders.
    """
    
    data_root = './data'
    if not os.path.exists(data_root):
        os.makedirs(data_root)

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

    download = not os.path.exists(os.path.join(data_root, dataset_name, 'train'))  # Check if data already exists

    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if run_type == 'short':
        num_train_samples = int(len(train_dataset) * short_run_fraction)
        num_test_samples = int(len(test_dataset) * short_run_fraction)
        train_indices = np.random.choice(len(train_dataset), num_train_samples, replace=False)
        test_indices = np.random.choice(len(test_dataset), num_test_samples, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
