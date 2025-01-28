import numpy as np


def calculate_velocity_and_theta(coords):
    """
    Calculate velocity and orientation (theta) between consecutive points.

    Args:
    coords: List of (x, y) coordinates

    Returns:
    velocities: List of velocities between points
    thetas: List of orientations (theta) between points
    """
    velocities = []
    thetas = []
    velocities.append(0)  # Assume initial velocity is 0
    thetas.append(0)  # Assume initial orientation is 0

    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        dx = x1 - x0
        dy = y1 - y0

        distance = np.sqrt(dx**2 + dy**2)
        velocities.append(distance)  # Assuming unit time interval for simplicity

        theta = np.arctan2(dy, dx)
        thetas.append(theta)

    return velocities, thetas


def downsample_path(coords, target_distance):
    """
    Downsample the path to have points approximately every target_distance meters apart.

    Args:
    coords: List of (x, y) coordinates
    target_distance: Desired distance between consecutive points in the reduced path

    Returns:
    reduced_coords: Downsampled list of coordinates
    """
    reduced_coords = [coords[0]]
    accumulated_distance = 0.0

    for i in range(1, len(coords)):
        x0, y0 = reduced_coords[-1]
        x1, y1 = coords[i]

        segment_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        accumulated_distance += segment_distance

        if accumulated_distance >= target_distance:
            reduced_coords.append((x1, y1))
            accumulated_distance = 0.0  # Reset for next segment

    # Ensure the last point is included
    if reduced_coords[-1] != coords[-1]:
        reduced_coords.append(coords[-1])

    return reduced_coords


# Example usage

coordinates_file = open("coords_mask_0.txt", "r")
original_coords = []
for line in coordinates_file:
    # each line contains co-ordinates x, y in format [x, y]
    # so need to remove [ and ] and split by comma
    x, y = line[1:-2].split(",")
    original_coords.append((float(x), float(y)))


# original_coords = [(0, 0), (1, 0.5), (2, 1), (3, 1.5), (4, 2), (5, 2.5), (6, 3)]
target_distance = 2.0  # meters

reduced_coords = downsample_path(original_coords, target_distance)
velocities, thetas = calculate_velocity_and_theta(reduced_coords)

# print("Velocities:", velocities)
# print("Orientations (Theta):", thetas)
# print("Reduced Path:", reduced_coords)

# write velocities, orientations and reduced path to files
# to their own seperate files but with prefix coords_mask_0
with open("coords_mask_0_velocities.txt", "w") as f:
    for item in velocities:
        f.write("%s\n" % item)

with open("coords_mask_0_orientations.txt", "w") as f:
    for item in thetas:
        f.write("%s\n" % item)

# for coords use [x, y] format
with open("coords_mask_0_reduced.txt", "w") as f:
    for item in reduced_coords:
        x, y = item
        f.write(f"[{x}, {y}]\n")

# create a csv file with the reduced path, velocities and orientations
import csv

with open("coords_mask_0.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "velocity", "orientation"])
    for i in range(len(reduced_coords)):
        x, y = reduced_coords[i]
        writer.writerow([x, y, velocities[i], thetas[i]])
