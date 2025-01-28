from scipy.signal import find_peaks
import cv2
import numpy as np


def find_dominant_colors(image_path, num_colors=4):
    """
    Finds the dominant colors in an image using color histograms.

    Args:
        image_path: Path to the image file.
        num_colors: Number of dominant colors to find.

    Returns:
        A list of dominant colors in BGR format.
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for each color channel
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

    # Find prominent peaks in the histograms
    h_peaks, _ = find_peaks(hist_h, height=hist_h.max() * 0.1)
    s_peaks, _ = find_peaks(hist_s, height=hist_s.max() * 0.1)
    v_peaks, _ = find_peaks(hist_v, height=hist_v.max() * 0.1)

    # Combine peaks and select top 'num_colors' combinations
    color_combinations = []
    for h in h_peaks:
        for s in s_peaks:
            for v in v_peaks:
                color_combinations.append((h, s, v))

    # Sort combinations based on peak heights (heuristic)
    sorted_combinations = sorted(
        color_combinations,
        key=lambda x: hist_h[x[0]] * hist_s[x[1]] * hist_v[x[2]],
        reverse=True,
    )

    # Select top 'num_colors' combinations
    selected_combinations = sorted_combinations[:num_colors]

    # Convert HSV back to BGR
    dominant_colors = []
    for h, s, v in selected_combinations:
        hsv_color = np.array([[h, s, v]])
        bgr_color = cv2.cvtColor(hsv_color.astype(np.uint8), cv2.COLOR_HSV2BGR)[0]
        dominant_colors.append(bgr_color)

    return dominant_colors


# Example usage
image_path = "test3.jpg"
dominant_colors = find_dominant_colors(image_path)

print("dominant color len ", len(dominant_colors))
# Display the dominant colors
for color in dominant_colors:
    print(f"Dominant Color: {color}")
