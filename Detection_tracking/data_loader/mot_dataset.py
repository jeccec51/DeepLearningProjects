"""Module to define MOTDataset class."""

import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class MOTDataset(Dataset):
    """
    MOTDataset class for loading frames and their corresponding annotations
    from the MOT20 dataset.

    Args:
        video_dir: Path to the directory containing video frames (img1 folders).
        annotation_dir: Path to the directory containing annotations (gt or det folders).
        sequence: Specify whether to use 'train' or 'test' sequences.
        use_gt: Whether to use ground truth (gt) or detection results (det) for annotations.
        transform: Transformations to be applied to the frames.
    """
    
    def __init__(self, video_dir: str, annotation_dir: str, sequence: str = 'train', use_gt: bool = True, transform=None):
        """Initializes the MOTDataset."""
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', video_dir))
        self.video_dir = os.path.join(base_dir, 'MOT20', sequence)
        self.annotation_dir = self.video_dir
        self.transform = transform
        self.use_gt = use_gt
        self.ann_folder = 'gt' if use_gt else 'det'

        self.sequences = [seq for seq in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, seq))]


    def __len__(self) -> int:
        """Returns the total number of frames across all sequences."""

        return sum([len(os.listdir(os.path.join(self.video_dir, seq, 'img1'))) for seq in self.sequences])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[Tuple[float, float, float, float]]]:
        """Retrieves a single frame and its annotations.

        Args:
            index: Index of the frame to retrieve.

        Returns:
            The image as a tensor, and a list of bounding boxes in (x_min, y_min, width, height) format.
        """

        cumulative_count = 0
        for sequence in self.sequences:
            img1_dir = os.path.join(self.video_dir, sequence, 'img1')
            img_files = sorted(os.listdir(img1_dir))
            num_frames = len(img_files)

            if index < cumulative_count + num_frames:
                img_file = img_files[index - cumulative_count]
                img_path = os.path.join(img1_dir, img_file)
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.transform:
                    frame = self.transform(frame)

                annotation_file = os.path.join(self.annotation_dir, sequence, f'{self.ann_folder}/{self.ann_folder}.txt')
                annotations = self._load_annotations(annotation_file, img_file)

                # Convert frame to tensor
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

                return frame_tensor, annotations

            cumulative_count += num_frames

        raise IndexError(f"Index {index} out of range")


    def _load_annotations(self, annotation_file: str, img_file: str) -> List[Tuple[float, float, float, float]]:
        """Loads annotations for a given image file.

        Args:
            annotation_file: Path to the annotation file.
            img_file: Name of the image file.

        Returns:
            A list of bounding boxes.
        """
        img_index = int(os.path.splitext(img_file)[0])
        annotations = []

        with open(annotation_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                frame_id = int(parts[0])

                if frame_id == img_index:
                    x_min = float(parts[2])
                    y_min = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    annotations.append((x_min, y_min, width, height))

        return annotations
