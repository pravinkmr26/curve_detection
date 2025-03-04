import cv2
import numpy as np
from scipy.interpolate import CubicSpline

def skeletonize(img):
    """Convert a binary image to a 1-pixel-wide skeleton."""
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton

def trace_skeleton(skeleton):
    """Trace the skeleton from one endpoint to another using BFS."""
    # Find endpoints (pixels with exactly one neighbor)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton, -1, kernel)
    endpoints = np.argwhere(filtered == 11)  # 1 neighbor (10 + 1)
    
    if len(endpoints) == 0:
        return None  # No endpoints found (closed loop)
    
    # Start from the first endpoint
    start = tuple(endpoints[0])
    visited = np.zeros_like(skeleton, dtype=bool)
    path = []
    
    # Define 8-directional movement
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    stack = [start]
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        path.append(node)
        # Explore neighbors
        neighbors = []
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            if 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1]:
                if skeleton[x, y] == 255 and not visited[x, y]:
                    neighbors.append((x, y))
        # Reverse to maintain order (LIFO)
        stack.extend(reversed(neighbors))
    
    return np.array(path)

def process_arrow(image_path):
    # Load and preprocess image (arrow white on black)
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Image not found")
        return None
    
    # Binarize and invert (arrow becomes white on black)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Skeletonize
    skeleton = skeletonize(binary)
    
    # Trace skeleton to get ordered points
    path = trace_skeleton(skeleton)
    if path is None or len(path) < 2:
        print("No valid path found")
        return None
    
    # Fit spline
    t = np.arange(len(path))
    cs_x = CubicSpline(t, path[:, 1])  # x = column (second index)
    cs_y = CubicSpline(t, path[:, 0])  # y = row (first index)
    t_fine = np.linspace(0, len(path)-1, 1000)
    x_fit = cs_x(t_fine)
    y_fit = cs_y(t_fine)
    fitted_curve = np.column_stack((x_fit, y_fit)).astype(np.int32)
    
    return fitted_curve

# Example usage
if __name__ == "__main__":
    curve = process_arrow("data/img2.png")
    
    if curve is not None:
        # Visualize
        img = cv2.imread(".png", 0)
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(result, [curve], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()