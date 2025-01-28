import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def detect_and_generate_color_masks(image_path, n_colors=5):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV (Hue, Saturation, Value)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array (each pixel is a point in HSV space)
    pixels = image_hsv.reshape(-1, 3)
    n_colors = n_colors + 1
    # Apply K-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors in HSV space)
    centers = kmeans.cluster_centers_.astype(int)

    # Get the labels for each pixel (which cluster each pixel belongs to)
    labels = kmeans.labels_

    # for label in labels:
    #     print(label)

    # Create masks and isolated images for each color cluster
    for i, center in enumerate(centers):
        # Define tolerance to create a range around the cluster center
        lower_bound = np.clip(center - 10, 0, 255)  # Lower tolerance
        upper_bound = np.clip(center + 10, 0, 255)  # Upper tolerance

        print("\n", center, lower_bound, upper_bound)

        # Create a mask for pixels in the cluster
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        # Bitwise AND to isolate the detected color
        result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

        # Display the result for each cluster
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Mask for Color {i+1}")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title(f"Color {i+1} Extracted")
        plt.show()


# Provide the image path
image_path = "test3.jpg"

# Detect and generate masks for 4-5 colors in the image
detect_and_generate_color_masks(image_path, n_colors=5)
