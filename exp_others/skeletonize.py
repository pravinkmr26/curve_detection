import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# Load the image
image_path = "test2_threshold.jpg"
image = cv2.imread(image_path, cv2.THRESH_BINARY)

# Skeletonize the image
skeleton = skeletonize(image)

# Convert skeleton to uint8 for visualization
skeleton_uint8 = (skeleton * 255).astype(np.uint8)

# Label connected components in the skeleton
labeled_skeleton, num_labels = label(skeleton, connectivity=2, return_num=True)

# Initialize list to store curve coordinates
curve_coordinates = []

# Extract coordinates for each labeled curve
for region in regionprops(labeled_skeleton):
    coords = region.coords  # Extract x, y coordinates
    curve_coordinates.append(coords)

# Visualize labeled curves
output_image = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2BGR)
for coords in curve_coordinates:
    for y, x in coords:
        output_image[y, x] = (0, 255, 0)

# Save the output visualization
cv2.imwrite("skeleton_curves.jpg", output_image)

# Save the coordinates
for i, coords in enumerate(curve_coordinates):
    print(coords)
    # np.save(f"skeleton_curve_{i+1}_coordinates.npy", coords)
    pass

print("Skeleton curves extracted and saved.")
