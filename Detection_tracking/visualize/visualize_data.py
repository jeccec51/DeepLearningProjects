"""Visualize Data Module."""

import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import List, Tuple
from data_loader.mot_dataset import MOTDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    for frame_index in tqdm(range(num_frames), desc="Visualizing Frames"):
        frame = frames_to_visualize[frame_index]

        # Check if the frame has 3 dimensions (C, H, W)
        if frame.dim() == 2:
            # Grayscale frame, add channel dimension
            frame = frame.unsqueeze(0)  # Convert (H, W) to (1, H, W)

        # Now frame should be (C, H, W)
        if frame.size(0) == 1:
            # If the frame is grayscale (single channel), repeat the channel to make it look like an RGB image
            frame = frame.repeat(3, 1, 1)

        # Convert from (C, H, W) to (H, W, C) for plotting
        frame = frame.permute(1, 2, 0).numpy()

        # Convert frame to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[-1] == 3 else frame

        # Plot frame with annotations
        plt.figure(figsize=(10, 10))
        plt.imshow(frame_rgb)

        # Draw bounding boxes
        if isinstance(frame_annotations, list):
            for box in frame_annotations:
                # Ensure the box has the expected four elements
                if isinstance(box, tuple) and len(box) == 4:
                    x_min, y_min, width, height = box
                    rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                else:
                    logging.error(f"Unexpected format in bounding box: {box}")
        else:
            logging.error(f"Unexpected format in frame_annotations: {frame_annotations}")

        plt.title(f"Frame {frame_index + 1} of sequence {sequence_index}")
        plt.axis('off')
        plt.show()

def plot_annotation_distribution(dataset: MOTDataset) -> None:
    """Plots the distribution of bounding box sizes and locations across the dataset.
    
    Args:
        dataset (MOTDataset): The dataset object.
    """
    bounding_box_widths = []
    bounding_box_heights = []
    aspect_ratios = []
    
    # Use tqdm to create a progress bar
    for sample_index in tqdm(range(len(dataset)), desc="Processing Bounding Boxes for Annotation Distribution"):
        _, annotations = dataset[sample_index]  # Get the annotations directly
        for box in annotations:
            if isinstance(box, tuple) and len(box) == 4:
                x_min, y_min, width, height = box
                bounding_box_widths.append(width)
                bounding_box_heights.append(height)
                if height > 0:
                    aspect_ratios.append(width / height)
            else:
                logging.error(f"Unexpected format in bounding box: {box}")
                raise ValueError("Bounding box is not in the expected format of (x_min, y_min, width, height).")

    # Convert lists to numpy arrays
    bounding_box_widths = np.array([w for w in bounding_box_widths if not (np.isnan(w) or np.isinf(w))])
    bounding_box_heights = np.array([h for h in bounding_box_heights if not (np.isnan(h) or np.isinf(h))])
    aspect_ratios = np.array([ar for ar in aspect_ratios if not (np.isnan(ar) or np.isinf(ar))])



    logging.info(f"Number of valid widths: {len(bounding_box_widths)}")
    logging.info(f"Number of valid heights: {len(bounding_box_heights)}")
    logging.info(f"Number of valid aspect ratios: {len(aspect_ratios)}")

    # Plotting distributions with the progress information
    plt.figure(figsize=(14, 6))

    # Plot the distribution of widths
    plt.subplot(1, 3, 1)
    plt.hist(bounding_box_widths, bins=30, color='blue', alpha=0.7)
    plt.title("Distribution of Bounding Box Widths (pixels)")
    plt.xlabel("Width")
    plt.ylabel("Frequency")

    # Plot the distribution of heights
    plt.subplot(1, 3, 2)
    plt.hist(bounding_box_heights, bins=30, color='green', alpha=0.7)
    plt.title("Distribution of Bounding Box Heights (pixels)")
    plt.xlabel("Height")
    plt.ylabel("Frequency")

    # Plot the distribution of aspect ratios
    plt.subplot(1, 3, 3)
    plt.hist(aspect_ratios, bins=100, color='red', alpha=0.7)
    plt.title("Distribution of Bounding Box Aspect Ratios (Width/Height)")
    plt.xlabel("Aspect Ratio (Width/Height)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_spatial_distribution(dataset: MOTDataset) -> None:
    """Plots the spatial distribution of bounding boxes across the dataset.
    
    Args:
        dataset (MOTDataset): The dataset object containing annotations.
    """
    x_centers = []
    y_centers = []

    for sample_index in tqdm(range(len(dataset)), desc="Processing Bounding Boxes for Spatial Distribution"):
        _, annotations = dataset[sample_index]
        for box in annotations:
            if isinstance(box, tuple) and len(box) == 4:
                x_min, y_min, width, height = box
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                x_centers.append(x_center)
                y_centers.append(y_center)
            else:
                logging.error(f"Unexpected format in bounding box: {box}")
                raise ValueError("Bounding box is not in the expected format of (x_min, y_min, width, height).")

    logging.info(f"Number of bounding box centers calculated: {len(x_centers)}")

    plt.figure(figsize=(10, 10))
    plt.hexbin(x_centers, y_centers, gridsize=50, cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.title("Spatial Distribution of Bounding Box Centers")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()
