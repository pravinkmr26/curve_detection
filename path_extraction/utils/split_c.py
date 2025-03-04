import numpy as np
import matplotlib.pyplot as plt

def compute_cumulative_distance(points):
    """Compute cumulative traversal distance along the looped curve."""
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return distances

def find_farthest_points_on_loop(points):
    """
    Finds the two farthest points on a closed curve using accumulated traversal distance.

    :param points: np.array of shape (N, 2), where each row is (x, y)
    :return: (index1, index2) of the two farthest points
    """
    points = np.array(points)  # Ensure it's a NumPy array

    # Step 1: Compute cumulative traversal distance along the curve
    distances = compute_cumulative_distance(points)

    # Step 2: Find total curve length
    total_length = distances[-1]

    # Step 3: Find the point where distance reaches half of total length
    half_distance = total_length / 2
    idx2 = np.searchsorted(distances, half_distance)  # Find the closest index

    # Start point is always the first point
    idx1 = 0  

    return idx1, idx2

# Example usage:
loop_points = np.array([
    [0, 0], [1, 2], [3, 4], [6, 5], [8, 4], [9, 2], [10, 0],  # Outward journey
    [9, -2], [8, -4], [6, -5], [3, -4], [1, -2], [0, 0]  # Returning back
])

idx1, idx2 = find_farthest_points_on_loop(loop_points)
print(f"Farthest points indices: {idx1}, {idx2}")
print(f"Farthest points: {loop_points[idx1]}, {loop_points[idx2]}")

# Plot the looped curve and mark the farthest points
plt.plot(loop_points[:, 0], loop_points[:, 1], 'bo-', label="Looped Curve")
plt.scatter([loop_points[idx1, 0], loop_points[idx2, 0]],
            [loop_points[idx1, 1], loop_points[idx2, 1]],
            color='red', s=100, label="Farthest Points")
plt.legend()
plt.title("Farthest Points on a Closed Curve")
plt.show()
