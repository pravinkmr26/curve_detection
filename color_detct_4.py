import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_all_colors(image_path, min_saturation=50, max_value=245, hue_tolerance=10):
    """
    Detect all distinct colors in an image, avoiding repetition of similar shades.

    Args:
    - image_path: Path to the image file.
    - min_saturation: Minimum saturation to exclude whites/grays.
    - max_value: Maximum value (brightness) to exclude light colors.
    - hue_tolerance: Tolerance to group nearby hues as one color.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Exclude white/gray pixels by masking based on Saturation and Value
    mask = (image_hsv[:, :, 1] > min_saturation) & (image_hsv[:, :, 2] < max_value)
    hue_values = image_hsv[:, :, 0][mask]

    # Compute the histogram for the Hue channel (fine-grained bins)
    hist, bin_edges = np.histogram(
        hue_values, bins=180, range=(0, 180)
    )  # 1-degree bins

    # Group similar hues dynamically
    detected_hues = []
    for i, count in enumerate(hist):
        if count > 0:
            # Check if this hue is far enough from existing clusters
            if not any(abs(i - h) <= hue_tolerance for h in detected_hues):
                detected_hues.append(i)

    print(f"Detected Distinct Colors: {len(detected_hues)}")

    # Create and visualize masks for each distinct color
    for i, hue in enumerate(detected_hues):
        # Define the HSV range for the current color
        lower_bound = np.array([hue - hue_tolerance, min_saturation, 50])
        upper_bound = np.array([hue + hue_tolerance, 255, max_value])

        # Create a mask for the current color
        color_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        # Apply the mask to the original image
        color_image = cv2.bitwise_and(image, image, mask=color_mask)

        # Visualize the result
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Color {i+1} (Hue: {hue})")
        plt.axis("off")
        plt.show()


# Test the function
image_path = "test3.jpg"  # Replace with your image path
detect_all_colors(image_path, min_saturation=50, max_value=245, hue_tolerance=10)
