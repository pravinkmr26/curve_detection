import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_colors_with_histogram(image_path, min_saturation=50, max_value=245, bins=36):
    """
    Detect dominant colors in an image using histograms on the Hue channel.

    Args:
    - image_path: Path to the image file.
    - min_saturation: Minimum saturation to exclude whites/grays.
    - max_value: Maximum value (brightness) to exclude light colors.
    - bins: Number of bins for the Hue histogram (controls granularity).
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

    # Compute the histogram for the Hue channel
    hist, bin_edges = np.histogram(hue_values, bins=bins, range=(0, 360))

    print("hist", hist)

    # Find peaks in the histogram
    dominant_hues = np.argsort(hist)[-5:]  # Get the indices of the top 5 peaks
    print("dominant", dominant_hues)
    dominant_hues = [((bin_edges[i] + bin_edges[i + 1]) / 2) for i in dominant_hues]

    print(f"Detected Dominant Hues: {dominant_hues}")

    # Visualize the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(
        bin_edges[:-1],
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        color="blue",
        alpha=0.7,
    )
    plt.title("Hue Histogram")
    plt.xlabel("Hue")
    plt.ylabel("Frequency")
    plt.show()

    # Generate masks for each dominant color
    for i, hue in enumerate(dominant_hues):
        lower_bound = np.array(
            [hue - 10, min_saturation, 50]
        )  # Define a range around the hue
        upper_bound = np.array([hue + 10, 255, max_value])
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

        # Apply the mask to the original image
        color_image = cv2.bitwise_and(image, image, mask=mask)

        # Display the result
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Extracted Color {i+1} (Hue: {hue:.2f})")
        plt.axis("off")
        plt.show()


# Test the function
image_path = "test3.jpg"  # Replace with your image path
detect_colors_with_histogram(image_path, min_saturation=50, max_value=245, bins=36)
