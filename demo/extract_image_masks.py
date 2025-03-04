import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

np.set_printoptions(threshold=np.inf)

def get_curve_masks(image_path : str, number_of_curves: int): 
# Read the image
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB

    # Filter out zero-value pixels (black color)
    non_black_pixels = image[np.any(image != [0, 0, 0], axis=-1)]

    # Reshape the non-black pixels to be a list of pixels
    pixels = non_black_pixels.reshape((-1, 3))

    # Use KMeans to find the dominant colors
    num_colors = number_of_curves  # Number of colors to identify
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the cluster centers (the dominant colors)
    colors = kmeans.cluster_centers_.astype(int)

    # Create masks for each color
    masks = []
    for color in colors:
        lower_bound = np.array(
            [max(c - 5, 0) if c != 0 else 0 for c in color], dtype=np.uint8
        )
        upper_bound = np.array(
            [min(c + 5, 255) if c != 0 else 0 for c in color], dtype=np.uint8
        )
        print(color, lower_bound, upper_bound)
        mask = cv.inRange(image, lower_bound, upper_bound)    
        masks.append(mask)
    return masks

