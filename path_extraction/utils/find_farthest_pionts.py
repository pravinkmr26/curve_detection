import numpy as np
import matplotlib.pyplot as plt

def compute_cumulative_distance(points):
    """Compute cumulative traversal distance along the curve"""
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return distances

def find_farthest_points_on_curve(points):
    """
    Finds the two farthest points along the curve using the traversal distance.

    :param points: np.array of shape (N, 2), where each row is (x, y)
    :return: (index1, index2) of the farthest points based on traversal distance
    """
    points = np.array(points)  # Ensure it's a NumPy array

    # Compute cumulative traversal distance
    distances = compute_cumulative_distance(points)

    # Find the two points with the max traversal distance
    idx1 = np.argmax(distances)  # The point farthest from start
    idx2 = 0  # Start with the first point

    max_distance = 0
    for i in range(len(points)):
        d = abs(distances[idx1] - distances[i])  # Traversal distance between i and idx1
        if d > max_distance:
            max_distance = d
            idx2 = i

    return min(idx1, idx2), max(idx1, idx2), max_distance  # Ensure correct ordering

# define main function
def main():
    # Example usage:
    curve_points = np.array([
        [0, 0], [1, 0.5], [2, 1], [3, 1.5], [4, 2], [5, 2.5],
        [6, 3], [7, 2.8], [8, 2.5], [9, 2], [10, 1.5], [11, 1], [12, 0.5]
    ])

    idx1, idx2 = find_farthest_points_on_curve(curve_points)
    print(f"Farthest points indices: {idx1}, {idx2}")
    print(f"Farthest points: {curve_points[idx1]}, {curve_points[idx2]}")

    # Plot the curve and mark the farthest points
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'bo-', label="Curve")
    plt.scatter([curve_points[idx1, 0], curve_points[idx2, 0]],
                [curve_points[idx1, 1], curve_points[idx2, 1]],
                color='red', s=100, label="Farthest Points")
    plt.legend()
    plt.title("Farthest Points Along a Curve")
    plt.show()

if __name__ == "__main__":
    main()
