"""Example code to sample a gaussian distribution."""

import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_samples(mean: float, std: float,
                         num_samples: int) -> np.ndarray:
    """Samples a fixed number froma gaussian distribution.

    Args:
        mean: Mean of gaussian distribution
        std : Standared Deviation of distribution
        num_samples: Desired number of samples

    Returns:
        list[int]: Nunber of gaussian distributed samples
    """

    return np.random.normal(loc=mean, scale=std, size=num_samples)


while True:
    print("\n Gaussian Sampler Menu")
    print("1. Generate Samples")
    print("2. Exit")
    choice = int(input("Enter the choice"))
    if choice == 1:
        try:
            mean_g = float(input("Enter the desired Mean: "))
            std_g = float(input("Enter the standared deviation for the distribution: "))
            num = int(input("Enter the number of desired samples: "))
            samples = get_gaussian_samples(mean=mean_g, std=std_g, num_samples=1000)
            plt.hist(samples, bins=50, color='skyblue')
            plt.title(f"Gaussian Distribution(mean={mean_g}, std={std_g})")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
        except ValueError:
            print("invalid input combinations. Please validate")
    elif choice == 2:
        print("Exiting...")
        break
    else:
        print("invalid")
