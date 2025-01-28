import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def detect_colors(image_path, min_saturation=50, max_value=245, tolerance=15):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of HSV values
    pixels = image_hsv.reshape(-1, 3)

    # Exclude white and very light pixels
    valid_pixels = pixels[(pixels[:, 1] > min_saturation) & (pixels[:, 2] < max_value)]

    # Quantize HSV values to group similar colors
    quantized_pixels = (valid_pixels // tolerance) * tolerance

    # Count unique colors
    color_counts = Counter(map(tuple, quantized_pixels))

    # Display the top detected colors
    print("Detected Colors (excluding white):")
    for color, count in color_counts.most_common():
        print(
            f"Color (Hue: {color[0]}, Saturation: {color[1]}, Value: {color[2]}) - Pixels: {count}"
        )

    # Visualize the colors
    visualize_colors(color_counts, tolerance)


def visualize_colors(color_counts, tolerance):
    # Create color patches for visualization
    for color, count in color_counts.most_common():
        patch = np.full((100, 100, 3), color, dtype=np.uint8)
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)
        plt.imshow(patch_bgr)
        plt.title(f"Pixels: {count}")
        plt.axis("off")
        plt.show()


# Test the function
image_path = "test3.jpg"  # Replace with your image path
detect_colors(image_path, min_saturation=50, max_value=245, tolerance=15)
