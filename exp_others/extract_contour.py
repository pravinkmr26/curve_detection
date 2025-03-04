import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread("mask0.png", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
# edges = cv2.Canny(image, threshold1=50, threshold2=150)

# Find contours in the image
contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Assume the largest contour is the arrow
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour for smoother points (optional)
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Sort the points in the contour to follow the arrow from start to end
# Use a custom sorting strategy or fit a curve, like:
# - Compute distance between points to infer direction.
# - Use principal component analysis (PCA) for ordered extraction.

# Example: Simple approach based on x-coordinates for demonstration
sorted_points = sorted(approximated_contour, key=lambda point: point[0][0])

# Extract and display the ordered (x, y) coordinates
ordered_coordinates = [(point[0][0], point[0][1]) for point in sorted_points]

# Print the ordered coordinates
for coord in ordered_coordinates:
    print(coord)

# Optional: Draw the ordered points on the image
f = open(f"coords_mask0.txt", "w")

for i, (x, y) in enumerate(ordered_coordinates):
    f.write(f"[{x}, {y}]\n")
    cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
    cv2.putText(
        image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
    )

# Display the image with ordered points
cv2.imshow("Ordered Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
