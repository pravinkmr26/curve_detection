import cv2 as cv
import time
import matplotlib.pyplot as plt

image = cv.imread("test5.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, threshold = cv.threshold(gray_image, 180, 255, cv.THRESH_BINARY_INV)

cv.imwrite("test2_threshold.jpg", threshold)

# tuple[Sequence[MatLike], MatLike]
contours, hierarchy = cv.findContours(threshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)


# paths = [cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True) for cnt in contours]
# for path in paths:
#     img = cv.drawContours(image, [path], -1, (0, 0, 255), 5)
#     cv.imshow("path", img)
#     pass

# for con in contours:
#     cv.imwrite("con.txt", con)
xs = []
ys = []
f = open("coords.txt", "w")
coords = []
for contour in contours:
    for position in contour:
        for pose in position:
            x, y = pose
            coords.append(pose)
            xs.append(x)
            ys.append(y)
            f.write(f"[{x}, {y}]\n")

# li = list(zip(xs, ys))
# plt.scatter(*zip(*li))
# # plt.plot(ys)
# plt.show()
# for i in range(len(contours)):
#     print(i)
#     contours_image = cv.drawContours(image, contours, i, (0, 0, 0), 3)
#     cv.imshow("Contours", contours_image)
#     cv.waitKey(0)
#     time.sleep(0.5)

# contours_image
# for path in paths:
#     img = cv.drawContours(image, [path], -1, (0, 0, 255), 5)
#     cv.imshow("path", img)
#     pass

# contours_image = cv.drawContours(image, contours, -1, (0, 0, 0), 3)
# cv.imshow("Contours", contours_image)
# cv.waitKey(0)
# time.sleep(0.5)

##################################################################
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components

import numpy as np


def compute_angle(p1, p2, p3):
    """
    Compute the angle (in degrees) between vectors p1->p2 and p2->p3.
    """
    v1 = p2 - p1
    v2 = p3 - p2

    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Dot product and angle calculation
    dot_product = np.dot(v1, v2)
    angle = np.arccos(
        np.clip(dot_product, -1.0, 1.0)
    )  # Clip to handle numerical instability
    return np.degrees(angle)


def is_path_valid_and_continous(previous_pose, current_pose, next_pose, dt, at):
    dist = np.linalg.norm(
        current_pose - previous_pose
    )  # Distance between consecutive points
    angle = compute_angle(
        previous_pose, current_pose, next_pose
    )  # Angle at the current point

    # print("angle", angle)
    return dist <= dt or angle <= at


def segment_by_distance_and_angle(
    coordinates, distance_threshold=5, angle_threshold=30
):
    """
    Segment coordinates into separate paths based on distance and angle thresholds.
    """
    path_index = 0
    paths = dict()  # initializes with 1000 elements
    paths[path_index] = [coordinates[0]]

    for i in range(1, len(coordinates) - 1):
        current_pose = coordinates[i]
        previous_pose = coordinates[i - 1]
        next_pose = coordinates[i + 1]
        if not is_path_valid_and_continous(
            previous_pose, current_pose, next_pose, distance_threshold, angle_threshold
        ):
            # Let's validate the other paths before creating a new one
            for idx in range(0, len(paths)):
                last_pose_of_a_path = paths[idx][-1]
                if is_path_valid_and_continous(
                    last_pose_of_a_path,
                    current_pose,
                    next_pose,
                    distance_threshold,
                    angle_threshold,
                ):
                    # Continue this path path
                    paths[idx].append(coordinates[i])
                    path_index = idx
                    break
            else:
                # Start a new path, : distance or angle threshold is exceeded
                path_index = path_index + 1
                paths[path_index] = [coordinates[i]]
        else:
            # continous path
            print("okay I can append here")
            paths[path_index].append(coordinates[i])
    return paths


# Segment paths with thresholds
paths = segment_by_distance_and_angle(coords, distance_threshold=2, angle_threshold=4)

# Output paths
# for i, path in enumerate(paths):
#     print(f"Path {i+1}: {path}")


# Plotting
plt.figure(figsize=(8, 6))
for key, path in paths.items():
    print("------", path)
    path = np.array(path)
    plt.scatter(path[:, 0], path[:, 1], marker="o")
    plt.title("Segmented Curves")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


############################################
def group_points(coordinates, distance_threshold=5):
    unvisited = set(range(len(coordinates)))
    paths = []

    while unvisited:
        # Start a new path
        current_path = []
        stack = [unvisited.pop()]  # Start with one unvisited point

        while stack:
            idx = stack.pop()
            current_path.append(coordinates[idx])

            # Find neighbors within the distance threshold
            to_visit = [
                i
                for i in unvisited
                if np.linalg.norm(coordinates[i] - coordinates[idx])
                < distance_threshold
            ]
            stack.extend(to_visit)
            unvisited -= set(to_visit)

        paths.append(np.array(current_path))

    return paths


# Example Usage


# paths = group_points(coords)

# # Plotting
# plt.figure(figsize=(8, 6))
# for path in paths:
#     plt.plot(path[:, 0], path[:, 1], marker="o")
# plt.title("Segmented 2")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.show()


######################################
from sklearn.cluster import DBSCAN

# Input: Array of coordinates


# # DBSCAN clustering
# dbscan = DBSCAN(
#     eps=5, min_samples=2
# )  # eps is the maximum distance between points in the same cluster
# labels = dbscan.fit_predict(coords)

# for label in labels:
#     print(label)

# # Group points by cluster
# paths = [coords[labels == i] for i in set(labels) if i != -1]  # -1 indicates noise

# # Output: Segregated paths
# for i, path in enumerate(paths):
#     print(f"Path {i+1}:")
#     print(path)
