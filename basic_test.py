import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# Define the data
X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

# Create a meshgrid from X and Y
X, Y = np.meshgrid(X, Y)

# Calculate Z values for each pair of X[i], Y[j]
Z = X * Y
R = X ** 2 + Y ** 2

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf_Z = ax.plot_surface(X, Y, Z, color="red", alpha=0.2, edgecolor='black')
surf_R = ax.plot_surface(X, Y, R, color="green", alpha=0.2, edgecolor='black')

# Calculate the color values for the data points based on Z values
min_z = Z.min()
max_z = Z.max()
colors_z = ['red' for _ in range(len(X.flatten()))]
for i, z_val in enumerate(Z.flatten()):
    normalized_z = (z_val - min_z) / (max_z - min_z)
    colors_z[i] = (normalized_z, 0, 0, 1)  # (R, G, B, alpha)

# Plot scatter markers for each data point with color based on Z values
sc_Z = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors_z, s=50)

# Calculate the color values for the data points based on Z values
min_r = R.min()
max_r = R.max()
colors_r = ['green' for _ in range(len(X.flatten()))]
for i, r_val in enumerate(R.flatten()):
    normalized_r = (r_val - min_r) / (max_r - min_r)
    colors_r[i] = (0, normalized_r, 0, 1)  # (R, G, B, alpha)

# Plot scatter markers for each data point with color based on Z values
sc_R = ax.scatter(X.flatten(), Y.flatten(), R.flatten(), c=colors_r, s=50)

ax.legend([surf_Z, surf_R], ['Z = X * Y', 'R = X^2 + Y^2'], loc='upper left')
# ax.legend([sc_Z, sc_R], ['Z = X * Y', 'R = X^2 + Y^2'], loc='upper right')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Values')

plt.show()
