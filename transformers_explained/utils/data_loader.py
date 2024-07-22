"""Data loader Module."""

from typing import Tuple
from ast import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(data_set_name: str, img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Get Data loaders for the specified dataset.
    
    Args:
        data_set_name: Name of the dataset
        img_size: Size of the image
        bath_size: Batcg Size
    
    Returns:
        Train and test Data loaders
    """

    transform = transforms.Compose([transforms.Resize((img_size, img_size), 
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                                        ))])
    if data_set_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown Dataset{data_set_name}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

