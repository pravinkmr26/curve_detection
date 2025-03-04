import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def extract_path_coordinates(image):
    # Read the image    
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold using Otsu's method (invert if the curve is dark)
    #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological closing to connect small gaps
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Extract coordinates of the curve pixels
    y_coords, x_coords = np.where(closed > 0)
    if len(x_coords) == 0:
        return [], []  # No pixels found
    
    points = np.column_stack((x_coords, y_coords))
    
    # Cluster points using DBSCAN to handle disconnects
    db = DBSCAN(eps=5, min_samples=10).fit(points)
    labels = db.labels_
    
    # Identify the largest cluster (excluding noise)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if not unique_labels.size:
        return [], []  # No clusters found
    main_cluster_label = unique_labels[np.argmax(counts)]
    main_points = points[labels == main_cluster_label]
    
    # Sort points along the primary axis using PCA
    pca = PCA(n_components=1)
    projected = pca.fit_transform(main_points)
    sorted_indices = np.argsort(projected.flatten())
    sorted_points = main_points[sorted_indices]
    
    # create np.array of x and y coordinates
    return np.array(sorted_points[:, 0], sorted_points[:, 1])    

# # Example usage
# x_coords, y_coords = extract_path_coordinates("path_image.png")
# print("X coordinates:", x_coords)
# print("Y coordinates:", y_coords)