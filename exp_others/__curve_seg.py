import cv2
import numpy as np

# Step 1: Load the image and preprocess
image_path = "test2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

# Invert binary image (optional, for better connectivity)
# binary = cv2.bitwise_not(binary)

# Step 2: Identify connected components (each curve as a separate component)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=4
)

# Step 3: Extract coordinates for each labeled curve
curve_coordinates = []

for label in range(1, num_labels):  # Skip background (label 0)
    # Create a mask for the current label
    label_mask = (labels == label).astype(np.uint8) * 255

    # Find non-zero (x, y) coordinates of this label (the curve)
    coords = np.column_stack(np.where(label_mask > 0))  # (y, x) format
    curve_coordinates.append(coords)

    # Optional: Visualize each curve separately
    isolated_curve = np.zeros_like(binary)
    isolated_curve[label_mask > 0] = 255
    cv2.imwrite(f"data/curve_{label}.jpg", isolated_curve)

# Step 4: Save curve coordinates to numpy arrays
for i, coords in enumerate(curve_coordinates):
    # np.save(f"/mnt/data/curve_{i+1}_coordinates.npy", coords)
    print(coords)
    pass

print(f"Extracted {len(curve_coordinates)} curves and saved coordinates.")
