import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv




def generate_gaussian_spacing(center, width, num_points):
  """Generates a deterministic Gaussian spacing of points around a center.

  Args:
    center: The center of the Gaussian distribution.
    width: The standard deviation of the Gaussian distribution.
    num_points: The number of points to generate.

  Returns:
    A numpy array of the Gaussian-spaced points.
  """

  # Generate Gaussian probability density function
  y=np.linspace(-width,width,num_points)
  x = erfinv(y)

  return center+x

# Example usage
center = 10
width = 50
num_points = 100

# Generate Gaussian-spaced points around the dip
positions = generate_gaussian_spacing(center, width, num_points)
print(len(positions))

# Scan the translation stage at each position and measure coincidence counts
# ... (rest of the code remains the same)

y=np.zeros(len(positions))
plt.figure()
plt.plot(positions, y, 'o')
plt.show()