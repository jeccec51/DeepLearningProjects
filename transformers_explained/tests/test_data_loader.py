""" Module to test data loader."""

import pytest
from torch.utils.data import DataLoader
from utils.data_loader import get_data_loaders

def test_get_data_loaders() -> None:
    """Test the data loader for CIFAR10 dataset."""
    # GIVEN a dataset name, image size, and batch size
    dataset_name = 'CIFAR10'
    img_size = 32
    batch_size = 4
    use_yuv = False
    run_type = "long"
    short_run_fraction = 0.1

    # WHEN loading the data
    train_loader, val_loader = get_data_loaders(dataset_name=dataset_name, img_size=img_size, batch_size=batch_size, 
                                                use_yuv=use_yuv, run_type=run_type, short_run_fraction=short_run_fraction)
    
    # THEN the data loaders should be correctly initialized
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader.dataset) == 50000
    assert len(val_loader.dataset) == 10000
    assert next(iter(train_loader))[0].shape == (4, 3, 32, 32)
