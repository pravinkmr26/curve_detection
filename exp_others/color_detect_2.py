import cv2
import numpy as np


def detect_colors_and_mask(image_path, num_colors=3):
    """
    Detects dominant colors in an image and creates masks for them.

    Args:
        image_path: Path to the image file.
        num_colors: Number of dominant colors to detect.

    Returns:
        A list of tuples, where each tuple contains:
            - The detected color in BGR format.
            - The corresponding mask.
    """

    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Reshape the image into a 2D array of pixels
    pixels = hsv.reshape((-1, 3))

    # Perform k-means clustering to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert cluster centers from HSV to BGR
    centers = centers.astype(np.uint8)
    colors = [tuple(center) for center in centers]

    # Create masks for each detected color
    masked_images = []
    for color in colors:
        lower_bound = np.array([color[0] - 10, 100, 100])
        upper_bound = np.array([color[0] + 10, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        masked_images.append((color, mask))

    return masked_images


# Example usage
image_path = "test3.jpg"
num_colors_to_detect = 5

color_masks = detect_colors_and_mask(image_path, num_colors_to_detect)

# Display the masked images
for color, mask in color_masks:
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(f"Color: {color}", masked_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
