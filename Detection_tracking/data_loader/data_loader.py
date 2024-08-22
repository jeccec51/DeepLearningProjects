"""Data Loader Module."""

from torch.utils.data import DataLoader, random_split
from data_loader.mot_dataset import MOTDataset, collate_fn
from torchvision import transforms
from typing import Tuple

# Define your data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_data_loaders(video_dir: str, annotation_dir: str, batch_size: int = 4, seq_len: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        video_dir: Path to the video directory.
        annotation_dir: Path to the annotation directory.
        batch_size: Number of samples per batch. Defaults to 4.
        seq_len: Number of frames in each sequence. Defaults to 8.
    
    Returns:
        Data loaders for training, validation, and testing.
    """

    dataset = MOTDataset(video_dir, annotation_dir, transform=transform, seq_len=seq_len)
    
    # Split the dataset into training, validation, and testing sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
   
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
