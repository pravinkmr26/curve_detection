import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def detect_and_generate_color_masks(image_path, n_colors=5):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to HSV (Hue, Saturation, Value)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of HSV values
    pixels = image_hsv.reshape(-1, 3)

    n_colors = 10

    # Reduce colors dynamically by rounding HSV values (color quantization)
    quantized_pixels = (
        np.round(pixels / 10) * 10
    )  # Reduce precision to group similar colors
    unique_colors, counts = np.unique(quantized_pixels, axis=0, return_counts=True)

    # Sort colors by frequency and select the top `n_colors`
    dominant_colors = unique_colors[np.argsort(-counts)][:n_colors]

    # Create masks and isolated images for each dominant color
    for i, color in enumerate(dominant_colors):
        # Define HSV range for the current color
        lower_bound = np.clip(color - 10, 0, 255)
        upper_bound = np.clip(color + 10, 0, 255)
        print(color, lower_bound, upper_bound)
        # Create a mask for the current color
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        # Apply the mask to the original image
        color_image = cv2.bitwise_and(image, image, mask=mask)

        # Display the results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Mask for Color {i+1}")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Extracted Color {i+1}")
        plt.show()


# Provide the image path
image_path = "test3.jpg"

# Detect and generate masks for the dominant colors in the image
detect_and_generate_color_masks(image_path, n_colors=5)
