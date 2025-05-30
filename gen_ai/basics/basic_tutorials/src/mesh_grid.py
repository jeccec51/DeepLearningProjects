"""Module describing the use of a meshgrid."""

import numpy as np
import matplotlib.pyplot as plt 

x_range = np.linspace(-2, 2, 5)
y_range = np.linspace(-2, 2, 5)

xx, yy = np.meshgrid(x_range, y_range)

plt.scatter(xx, yy, c='blue')
plt.title("Meshgrid Points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

print("xx: \n", xx)
print("yy: \n", yy)
