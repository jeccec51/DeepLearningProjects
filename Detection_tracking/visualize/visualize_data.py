"""Visualize Data Module."""

import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from data_loader.mot_dataset import MOTDataset


def visualize_single_frame(image_path: str, bounding_boxes: List[Tuple[float, float, float, float]]) -> None:
    """Visualizes a single image with its bounding box annotations.
    
    Args:
        image_path: Path to the image file.
        bounding_boxes: List of bounding boxes in (x_min, y_min, width, height) format.
    """

    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Plot image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    # Draw bounding boxes
    for box in bounding_boxes:
        x_min, y_min, width, height = box
        rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.title(f"Annotated Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()


def visualize_frame_sequence(dataset: MOTDataset, sequence_index: int = 0, num_frames: int = 10) -> None:
    """Visualizes a sequence of frames from the dataset with annotations.
    
    Args:
        dataset: The dataset object.
        sequence_index: Index of the video sequence to visualize.
        num_frames: Number of frames to visualize.
    """

    frames, frame_annotations = dataset[sequence_index]
    frames_to_visualize = frames[:num_frames]  # Limit to the desired number of frames

    for frame_index in range(num_frames):
        frame = frames_to_visualize[frame_index].permute(1, 2, 0).numpy()  # Convert from tensor (C, H, W) to (H, W, C)
        bounding_boxes = frame_annotations[frame_index]
        
        # Convert frame to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Plot frame with annotations
        plt.figure(figsize=(10, 10))
        plt.imshow(frame_rgb)
        
        # Draw bounding boxes
        for box in bounding_boxes:
            x_min, y_min, width, height = box
            rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        
        plt.title(f"Frame {frame_index+1} of sequence {sequence_index}")
        plt.axis('off')
        plt.show()


def plot_annotation_distribution(dataset: MOTDataset) -> None:
    """Plots the distribution of bounding box sizes and locations across the dataset.
    
    Args:
        dataset: The dataset object containing annotations.
    """
    bounding_box_widths = []
    bounding_box_heights = []
    aspect_ratios = []
    for sample_index in range(len(dataset)):
        _, annotations = dataset[sample_index]
        for bounding_boxes in annotations:
            for box in bounding_boxes:
                x_min, y_min, width, height = box
                bounding_box_widths.append(width)
                bounding_box_heights.append(height)
                aspect_ratios.append(width / height)

    plt.figure(figsize=(14, 6))

    # Plot the distribution of widths
    plt.subplot(1, 3, 1)
    sns.histplot(bounding_box_widths, kde=True, bins=30)
    plt.title("Distribution of Bounding Box Widths")
    plt.xlabel("Width")

    # Plot the distribution of heights
    plt.subplot(1, 3, 2)
    sns.histplot(bounding_box_heights, kde=True, bins=30)
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
        dataset (MOTDataset): The dataset object containing annotations.
    """
    x_centers = []
    y_centers = []
    for sample_index in range(len(dataset)):
        _, annotations = dataset[sample_index]
        for bounding_boxes in annotations:
            for box in bounding_boxes:
                x_min, y_min, width, height = box
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                x_centers.append(x_center)
                y_centers.append(y_center)

    plt.figure(figsize=(10, 10))
    plt.hexbin(x_centers, y_centers, gridsize=50, cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.title("Spatial Distribution of Bounding Box Centers")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()

