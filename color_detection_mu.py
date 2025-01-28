import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Function to detect and generate masks for each color
def detect_and_generate_color_masks(image_path, n_colors=5):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels (each pixel is a point in RGB space)
    pixels = image_rgb.reshape(-1, 3)

    # Apply K-means clustering to find the most dominant colors in the image
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors)
    centers = np.uint8(kmeans.cluster_centers_)

    # Get the labels for each pixel (which cluster each pixel belongs to)
    labels = kmeans.labels_

    # Create a mask for each color cluster and generate an image for each color
    for i, center in enumerate(centers):
        print(center)
        # Create a mask where the pixel belongs to the current cluster
        mask = (labels == i).reshape(image_rgb.shape[0], image_rgb.shape[1])

        # Create an image where only the current color is visible
        color_image = np.zeros_like(image_rgb)
        color_image[mask] = center

        # Display the mask and the color image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Mask for Color {i+1} - {center}")
        plt.subplot(1, 2, 2)
        plt.imshow(color_image)
        plt.title(f"Color {i+1} Extracted")
        plt.show()
        # cv2.waitKey(0)


# Provide the image path
image_path = "test3.jpg"

# Detect and generate masks for dominant colors in the image
detect_and_generate_color_masks(image_path, n_colors=5)
