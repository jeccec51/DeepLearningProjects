"""Main Visu module."""

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
        video_dir=cfg.data_loader.video_dir,
        annotation_dir=cfg.data_loader.video_dir,
    )

    # Check if a short run is enabled and adjust the dataset size accordingly
    if cfg.visu.short_run:
        total_samples = len(dataset)
        subset_size = int(total_samples * cfg.visu.short_run_percentage)
        indices = random.sample(range(total_samples), subset_size)
        dataset = [dataset[idx] for idx in indices]
        print(f"Short run enabled: Using {len(dataset)} out of {total_samples} samples for visualization.")

    # Plot annotation distribution if enabled in the config
    if cfg.visu.plot_annotation_distribution:
        print("Plotting annotation distribution across the selected dataset...")
        plot_annotation_distribution(dataset)

    # Plot spatial distribution if enabled in the config
    if cfg.visu.plot_spatial_distribution:
        print("Plotting spatial distribution of bounding box centers across the selected dataset...")
        plot_spatial_distribution(dataset)

    # Visualize frame sequence if enabled in the config
    if cfg.visu.visualize_frame_sequence:
        random_index = random.randint(0, len(dataset) - 1)
        print(f"Visualizing sequence {random_index} from the dataset...")
        visualize_frame_sequence(dataset, sequence_index=random_index, num_frames=5)

    # Visualize a single random frame if enabled in the config
    if cfg.visu.visualize_single_frame:
        random_index = random.randint(0, len(dataset) - 1)
        _, annotations = dataset[random_index]
        image_path = f"{cfg.data_loader.video_dir}/img1/{random_index}.jpg"  # Assuming you have a valid path or adjust accordingly
        print(f"Visualizing a single frame from the dataset: Frame {random_index}")
        visualize_single_frame(image_path, annotations)

if __name__ == "__main__":
    main()
