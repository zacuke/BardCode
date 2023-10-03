import numpy as np
import matplotlib.pyplot as plt

def u(z):
  """A subharmonic exhaustion function."""
  return -2 * np.log(np.abs(1 - np.exp(1j * z)))

def H2_plus(z):
  """The Poletsky-Stessin Hardy space H^2_+(D)."""
  integral = 0
  for n in range(200):
    integral += np.abs(z**n) * np.exp(-u(z))
  return integral

# Create a 2D mesh of points in the complex plane.
X, Y = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000))
Z = X + 1j * Y

# Compute the Poletsky-Stessin Hardy space at each point in the mesh.
H2_plus_values = H2_plus(Z)

# Plot the Poletsky-Stessin Hardy space.
plt.contourf(X, Y, H2_plus_values, cmap='viridis')
plt.colorbar()
plt.title('Poletsky-Stessin Hardy space H^2_+(D)')
plt.show(block=True)