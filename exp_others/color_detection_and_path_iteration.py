import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from curve_detection_by_skeletonizing import extract_contours_and_write_coords_to_file

np.set_printoptions(threshold=np.inf)

# Read the image
image = cv.imread("test2.png", cv.IMREAD_COLOR)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB
# gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Filter out zero-value pixels (black color)
non_black_pixels = image[np.any(image != [0, 0, 0], axis=-1)]

# Reshape the non-black pixels to be a list of pixels
pixels = non_black_pixels.reshape((-1, 3))

# Use KMeans to find the dominant colors
num_colors = 3  # Number of colors to identify
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(pixels)

# Get the cluster centers (the dominant colors)
colors = kmeans.cluster_centers_.astype(int)

# Create masks for each color
masks = []
for color in colors:
    lower_bound = np.array(
        [max(c - 5, 0) if c != 0 else 0 for c in color], dtype=np.uint8
    )
    upper_bound = np.array(
        [min(c + 5, 255) if c != 0 else 0 for c in color], dtype=np.uint8
    )
    print(color, lower_bound, upper_bound)
    mask = cv.inRange(image, lower_bound, upper_bound)
    # print(image.type())
    # cv.imshow("found path", mask)
    # cv.waitKey(0)
    masks.append(mask)

# # Find non-zero pixels for each mask
# non_zero_pixels = [cv.findNonZero(mask) for mask in masks]

# Print the colors and their corresponding non-zero pixels
for i, color in enumerate(colors):
    print(f"Color {i + 1}: {color}")
    # f = open(f"color_{i}.txt", "w")
    # f.write(f"Non-zero pixels for color {i + 1}: {non_zero_pixels[i]}")
    # f.close()

# Optionally, you can visualize the masks
for i, mask in enumerate(masks):
    extract_contours_and_write_coords_to_file(mask, f"coords_mask_{i}.txt")
# cv.imshow("black image", gray_image)
cv.imshow("final image", image)
cv.waitKey(0)
cv.destroyAllWindows()
