"""Module to mimic a basic regressor."""

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression


def generate_2d_points_label_pair() -> tuple[ndarray, ndarray]:
    """
    Generates a pair of 2D points and their corresponding labels.

    Returns:
        tuple (ndarray): A tuple containing two numpy arrays:
            - The first array contains the generated 2D points with shape.
            - The second array contains the labels for each point with shape.

    Raises:
        None

    Example:
        points, labels = generate_2d_points_label_pair()
    """

    result = make_blobs(n_samples=500, centers=2,
                        cluster_std=1.0, random_state=42)

    points_, labels_ = result[:2]
    return points_, labels_


# Plot points
points, labels = generate_2d_points_label_pair()

# Create Logistic Regression classifier
clf = LogisticRegression()
clf.fit(points, labels)

# Create Mesh Grid for visualization
x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Predict the class
grid_points = np.c_[xx.ravel(), yy.ravel()]
pred = clf.predict(grid_points)
pred = pred.reshape(xx.shape)

# Plot the predictions

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, pred, alpha=0.4, cmap='coolwarm')
plt.scatter(points[:,0], points[:,1], c=labels, edgecolors='k', cmap='coolwarm')
plt.title("Logistic Regression-Discriminative Model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
