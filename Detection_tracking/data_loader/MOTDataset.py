"""Module to define MOTDataset class."""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Callable, Tuple, List, Optional


class MOTDataset(Dataset):
    def __init__(self, video_dir: str, annotation_dir: str, transform: Optional[Callable] = None, seq_len: int = 8) -> None:
        """Dataset for the MOT challenge videos and annotations.
        
        Args:
            video_dir: Directory containing the video files.
            annotation_dir: Directory containing the annotation files.
            transform: Optional transform to be applied on a sample. Defaults to None.
            seq_len: Number of frames in each sequence. Defaults to 8.
        """

        super().__init__()  # Ensures proper initialization of the parent Dataset class
        
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.seq_len = seq_len
        self.video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')])
        self.annotation_files = sorted([os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.txt')])


    def __len__(self) -> int:
        """Returns the number of video files in the dataset."""
        
        return len(self.video_files)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sequence of frames and corresponding annotations from the dataset.
        
        Args:
            idx: Index of the video file to load.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the stacked frames and their corresponding annotations.
        """

        video_path = self.video_files[idx]
        annotation_path = self.annotation_files[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        ret, frame = cap.read()
        frame_count = 0
        while ret and frame_count < self.seq_len:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            ret, frame = cap.read()
            frame_count += 1
        cap.release()
        
        annotations = self.load_annotations(annotation_path, frame_count)
        
        return torch.stack(frames), annotations


    def load_annotations(self, annotation_path: str, frame_count: int) -> torch.Tensor:
        """Load annotations from the annotation file and align them with the frames.
        
        Args:
            annotation_path: Path to the annotation file.
            frame_count: Number of frames loaded.
        
        Returns:
            torch.Tensor: Aligned annotations as a tensor.
        """
        
        annotations = []
        with open(annotation_path, 'r') as file:
            for line in file:
                values = line.strip().split(',')
                frame_idx = int(values[0])
                if frame_idx < frame_count:
                    bbox = [float(values[2]), float(values[3]), float(values[4]), float(values[5])]
                    annotations.append(bbox)
        return torch.tensor(annotations, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Custom collate function to handle batches of frames and annotations.
    
    Args:
        batch: Batch of samples.
    
    Returns:
        A tuple containing the stacked frames and a list of annotations.
    """
    
    frames, annotations = zip(*batch)
    return torch.stack(frames), list(annotations)
