import cv2
import numpy as np


def extract_contours_and_write_coords_to_file(image: cv2.typing.MatLike, filename):
    # Convert the image to grayscale
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours in the skeletonized image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    paths = []
    for contour in contours:
        # Select the longest contour assuming it represents the main line
        # longest_contour = max(contours, key=cv2.contourArea)

        # Reshape the contour to a list of (x, y) points
        contour_points = contour.reshape(-1, 2)

        # If the contour is a loop, break the loop to form a line-like structure
        # We assume the endpoints are the points with maximum distance from each other
        distances = np.linalg.norm(contour_points - contour_points[:, None], axis=2)
        start_idx, end_idx = np.unravel_index(distances.argmax(), distances.shape)
        if start_idx > end_idx:
            contour_points = contour_points[start_idx:end_idx:-1]
        else:
            contour_points = contour_points[start_idx : end_idx + 1]

        paths.append(contour_points)

    def get_distance(path1, path2):
        return np.linalg.norm(path1 - path2)

    # # Sort the paths based on the distance between the end of one path and the start of the next
    # do this for all paths, they can be in any order
    print("total paths", len(paths))
    final_path = paths[0]
    paths.pop(0)
    threshold = 10
    while len(paths) > 0:
        for i, path in enumerate(paths):
            start_to_end = get_distance(final_path[0], path[-1])
            end_to_start = get_distance(final_path[-1], path[0])
            start_to_start = get_distance(final_path[0], path[0])
            end_to_end = get_distance(final_path[-1], path[-1])
            print("merging contour", i)
            if start_to_end < threshold:
                print("start to end", start_to_end)
                final_path = np.concatenate((path, final_path))
                paths.pop(i)
                break
            elif end_to_start < threshold:
                print("end to start", end_to_start)
                final_path = np.concatenate((final_path, path))
                paths.pop(i)
                break
            elif start_to_start < threshold:
                print("start to start", start_to_start)
                final_path = np.concatenate((path[::-1], final_path))
                paths.pop(i)
                break
            elif end_to_end < threshold:
                print("end to end", end_to_end)
                final_path = np.concatenate((final_path, path[::-1]))
                paths.pop(i)
                break
            else:
                print("No match found")

    # Print the ordered coordinates
    f = open(filename, "w")
    for i, point in enumerate(final_path):
        # print("iter", i)
        x, y = point
        f.write(f"[{x}, {y}]\n")
    f.close()


# # Optional: Visualize the contour
# output_image = np.zeros_like(skeleton)
# for point in contour_points:
#     cv2.circle(output_image, tuple(point), 1, 255, -1)

# cv2.imshow("Ordered Path", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

extract_contours_and_write_coords_to_file(cv2.imread("mask0.png"), "mask0_coords.txt")
