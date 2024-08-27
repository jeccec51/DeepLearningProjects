"""Module to define MOTDataset class."""

import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class MOTDataset(Dataset):
    """Mot Data set Class.

        Args:
            video_dir: Path to the directory containing video frames (img1 folders).
            annotation_dir: Path to the directory containing annotations (det or gt folders).
            sequence: Specify whether to use 'train' or 'test' sequences.
            transform: Transformations to be applied to the frames.        
    """
    
    def __init__(self, video_dir: str, annotation_dir: str, sequence: str = 'train', transform=None):
        """Initializes the MOTDataset. """

        
        self.video_dir = os.path.join(video_dir, 'MOT20', sequence)
        self.annotation_dir = os.path.join(annotation_dir, 'MOT20', sequence)
        self.transform = transform

        self.sequences = [seq for seq in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, seq))]


    def __len__(self):
        return sum([len(os.listdir(os.path.join(self.video_dir, seq, 'img1'))) for seq in self.sequences])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[Tuple[float, float, float, float]]]:
        """Retrieves a single frame and its annotations.

        Args:
            index : Index of the frame to retrieve.

        Returns:
            The image as a tensor.
            List of bounding boxes in (x_min, y_min, width, height) format.
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

                annotation_file = os.path.join(self.annotation_dir, sequence, 'gt/gt.txt')
                annotations = self._load_annotations(annotation_file, img_file)

                return frame, annotations

            cumulative_count += num_frames

        raise IndexError(f"Index {index} out of range")
    

    def _load_annotations(self, annotation_file: str, img_file: str) -> List[Tuple[float, float, float, float]]:
        """Loads annotations for a given image file.

        Args:
            annotation_file: Path to the annotation file.
            img_file: Name of the image file.

        Returns:
            List of bounding boxes.
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
