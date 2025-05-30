"""Basic generator module. """


import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from basics.src.tutorials.basic_classifier import generate_2d_points_label_pair


def estimate_mean_std(point_coords: ndarray,
                      point_labels: ndarray
                      ) -> dict[int, tuple[ndarray, ndarray]]:
    """Module to estimate mean and standared deviation of each class.

    Args:
        point_coords: 2D coordinates of all the points
        point_labels : Labels of all the point

    Returns:
        A dictionary mapping each class to its mean and std.
    """

    class_stats: dict[int, tuple[ndarray, ndarray]] = {}
    for class_id in np.unique(point_labels):
        class_points = point_coords[point_labels == class_id]
        mean = class_points.mean(axis=0)
        std = class_points.std(axis=0)
        class_stats[class_id] = (mean, std)

    return class_stats


def sample_from_class_distributions(
    class_stats: dict[int, tuple[ndarray, ndarray]],
    num_samples_per_class: int = 250
) -> tuple[ndarray, ndarray]:
    """Class to sample from the generated points.

    Args:
        class_stats: Dict containing the class statistics 
        num_samples_per_class: Samples to be generated per class

    Returns:
        tuple of generated points and their blobs.
    """

    generated_pts = []
    generated_labs = []

    for class_id, (mean, std) in class_stats.items():
        smaples = np.random.normal(
            loc=mean, scale=std, size=(num_samples_per_class, 2)
        )
        generated_pts.append(smaples)
        generated_labs.extend([class_id] * num_samples_per_class)

    return np.vstack(generated_pts), np.array(generated_labs)


def visualize_generated_vs_real(
    original: tuple[ndarray, ndarray],
    generated: tuple[ndarray, ndarray]
) -> None:
    """Visu for generated points.

    Args:
        original: Original points
        generated: Generated Points
    """

    real_points, real_labels = original
    generated_pts, generated_labs = generated

    plt.figure(figsize=(8, 6))
    plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels,
                cmap='coolwarm', label='Original',
                edgecolor='k', alpha=0.6)
    plt.scatter(generated_pts[:, 0], generated_pts[:, 1],
                c=generated_labs,
                cmap='coolwarm', marker='x', label='Generated')
    plt.title("Generative Model: Class-wise Gaussian Sampling")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


points, labels = generate_2d_points_label_pair()
class_distributions = estimate_mean_std(point_coords=points,
                                        point_labels=labels)
generated_points, generated_labels = sample_from_class_distributions(
    class_stats=class_distributions,
    num_samples_per_class=500)
visualize_generated_vs_real(original=(points, labels),
                            generated=(generated_points, generated_labels))
