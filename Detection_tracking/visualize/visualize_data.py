"""Visualize Data Module."""

import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from data_loader.mot_dataset import MOTDataset


def visualize_single_frame(image_path: str, annotation: List[Tuple[float, float, float, float]]) -> None:
    """Visualizes a single image with its bounding box annotations.
    
    Args:
        image_path: Path to the image file.
        annotation: List of bounding boxes in (x, y, w, h) format.
    """

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Plot image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw bounding boxes
    for bbox in annotation:
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.title(f"Annotated Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()


def visualize_frame_sequence(dataset: MOTDataset, sequence_idx: int = 0, num_frames: int = 10) -> None:
    """Visualizes a sequence of frames from the dataset with annotations.
    
    Args:
        dataset: The dataset object.
        sequence_idx: Index of the video sequence to visualize.
        num_frames: Number of frames to visualize.
    """

    frames, annotations = dataset[sequence_idx]
    frames = frames[:num_frames]  # Limit to the desired number of frames

    for index in range(num_frames):
        frame = frames[index].permute(1, 2, 0).numpy()  # Convert from tensor (C, H, W) to (H, W, C)
        annotation = annotations[index]
        
        # Convert frame to RGB for Matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Plot frame with annotations
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        
        # Draw bounding boxes
        for bbox in annotation:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        
        plt.title(f"Frame {i+1} of sequence {sequence_idx}")
        plt.axis('off')
        plt.show()


def plot_annotation_distribution(dataset: MOTDataset) -> None:
    """Plots the distribution of bounding box sizes and locations across the dataset.
    
    Args:
        dataset: The dataset object containing annotations.
    """

    widths = []
    heights = []
    aspect_ratios = []
    for index in range(len(dataset)):
        _, annotations = dataset[index]
        for annotation in annotations:
            for bbox in annotation:
                x, y, w, h = bbox
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)

    plt.figure(figsize=(14, 6))

    # Plot the distribution of widths
    plt.subplot(1, 3, 1)
    sns.histplot(widths, kde=True, bins=30)
    plt.title("Distribution of Bounding Box Widths")
    plt.xlabel("Width")

    # Plot the distribution of heights
    plt.subplot(1, 3, 2)
    sns.histplot(heights, kde=True, bins=30)
    plt.title("Distribution of Bounding Box Heights")
    plt.xlabel("Height")

    # Plot the distribution of aspect ratios
    plt.subplot(1, 3, 3)
    sns.histplot(aspect_ratios, kde=True, bins=30)
    plt.title("Distribution of Bounding Box Aspect Ratios")
    plt.xlabel("Aspect Ratio (Width/Height)")

    plt.tight_layout()
    plt.show()


def plot_spatial_distribution(dataset: MOTDataset) -> None:
    """Plots the spatial distribution of bounding boxes across the dataset.
    
    Args:
        dataset: The dataset object containing annotations.
    """

    x_centers = []
    y_centers = []
    for index in range(len(dataset)):
        _, annotations = dataset[index]
        for annotation in annotations:
            for bbox in annotation:
                x, y, w, h = bbox
                x_centers.append(x + w / 2)
                y_centers.append(y + h / 2)

    plt.figure(figsize=(10, 10))
    plt.hexbin(x_centers, y_centers, gridsize=50, cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.title("Spatial Distribution of Bounding Box Centers")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()
