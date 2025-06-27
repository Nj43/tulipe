import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data: list of 3D points (to minimize all objectives)
points = np.random.rand(100, 3)

def is_pareto_efficient(points):
    """
    Identify Pareto-efficient points.
    Assumes minimization of all objectives.
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1) | np.all(points[is_efficient] == c, axis=1)
            is_efficient[i] = True  # keep self
    return is_efficient

# Get Pareto points
pareto_mask = is_pareto_efficient(points)
pareto_points = points[pareto_mask]
non_pareto_points = points[~pareto_mask]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Pareto points
ax.scatter(*pareto_points.T, c='red', label='Pareto Front')
# Plot others
ax.scatter(*non_pareto_points.T, c='blue', alpha=0.3, label='Non-Pareto')

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.legend()
plt.title('3D Pareto Frontier')
plt.show()