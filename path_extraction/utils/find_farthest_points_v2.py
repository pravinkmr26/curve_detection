import numpy as np

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

    return idx1, idx2, total_length