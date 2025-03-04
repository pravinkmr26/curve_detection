import cv2
import numpy as np


def detect_colored_lines(image_path, line_colors):
    """
    Detects thin colored lines in an image with a mostly white background.

    Args:
        image_path: Path to the image file.
        line_colors: A list of colors to detect in BGR format (e.g., [(0, 0, 255), (0, 255, 0)] for red and green).

    Returns:
        A list of masks, where each mask corresponds to a detected color.
    """

    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = []
    for color in line_colors:
        # Define color ranges in HSV for each line color
        lower_bound = np.array([color[0] - 10, 50, 50])  # Adjust tolerance as needed
        upper_bound = np.array([color[0] + 10, 255, 255])

        # Create a mask for the current color
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # Thin the mask to isolate lines (optional)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        masks.append(mask)

    return masks


# Example usage
image_path = "test3.jpg"
line_colors = [(0, 0, 255), (0, 255, 0)]  # Red and green lines

detected_lines = detect_colored_lines(image_path, line_colors)

# Display the results (optional)
for i, mask in enumerate(detected_lines):
    cv2.imshow(f"Color_{i+1}_Lines", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
