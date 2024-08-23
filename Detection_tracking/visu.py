"""Main Visu module. """

import hydra
from omegaconf import DictConfig
import random
from data_loader.mot_dataset import MOTDataset
from visualize.visualize_data import (
    visualize_single_frame,
    visualize_frame_sequence,
    plot_annotation_distribution,
    plot_spatial_distribution
)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function for visualizing data samples and distributions.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
 
    # Initialize dataset using the paths from the Hydra config
    dataset = MOTDataset(
        video_dir=cfg.data_loader.data_loader.video_dir,
        annotation_dir=cfg.data_loader.data_loader.annotation_dir,
    )

    # Visualize the annotation distribution across the entire dataset
    print("Plotting annotation distribution across the entire dataset...")
    plot_annotation_distribution(dataset)
    
    # Visualize the spatial distribution of bounding boxes across the entire dataset
    print("Plotting spatial distribution of bounding box centers across the entire dataset...")
    plot_spatial_distribution(dataset)

    # Randomly select a sample from the dataset for frame and sequence visualization
    random_index = random.randint(0, len(dataset) - 1)
    
    # Visualize a random frame sequence with annotations
    print(f"Visualizing sequence {random_index} from the dataset...")
    visualize_frame_sequence(dataset, sequence_index=random_index, num_frames=5)

if __name__ == "__main__":
    main()
