import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import distance_matrix, distance
from scipy.spatial.distance import cdist

from scipy.interpolate import CubicSpline
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
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

def process_arrow(binary):
    # # Load and preprocess image (arrow white on black)
    # img = cv2.imread(image_path, 0)
    # if img is None:
    #     print("Image not found")
    #     return None
    
    # # Binarize and invert (arrow becomes white on black)
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
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

# # Example usage
# if __name__ == "__main__":
#     curve = process_arrow_contours("complex_arrow.png", max_gap=20)
    
#     # Visualize
#     image = cv2.imread("complex_arrow.png", 0)
#     result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(result, [curve], isClosed=False, color=(0, 0, 255), thickness=2)
#     cv2.imshow("Result", result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
     
def extract_curve_from_mask(image: cv2.typing.MatLike):
    # Convert the image to grayscale
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours in the skeletonized image
    #contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    # draw contours with color blue and thickness 5
    # cv2.drawContours(bgr, contours, -1, (255, 255, 0), 2)
    # cv2.imshow("contours", bgr)
    # cv2.waitKey(0)

    # find non-zero pixels and get x, y coordinates
    #contours = cv2.findNonZero(binary_image)

    paths = []
    # for contour in contours:
    #     # Reshape the contour to a list of (x, y) points
    #     contour_points = contour.reshape(-1, 2)
    #     print("contour_points", contour_points)

    #     # If the contour is a loop, break the loop to form a line-like structure
    #     # We assume the endpoints are the points with maximum distance from each other
    #     distances = np.linalg.norm(contour_points - contour_points[:, None], axis=2)
    #     # this segment does not work as expected
    #     # fix this now
    #     start_idx, end_idx = np.unravel_index(distances.argmax(), distances.shape)
    #     print("start_idx, end_idx", start_idx, end_idx)
    #     if start_idx > end_idx:
    #         contour_points = contour_points[start_idx:end_idx:-1]
    #     else:
    #         contour_points = contour_points[start_idx : end_idx + 1]

    #     paths.append(contour_points)
    curve = process_arrow(binary_image)
    
    # Visualize
    # image = cv2.imread("arrow.png", 0)
    # result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # for curve in paths:
    #     cv2.polylines(bgr, [curve], isClosed=False, color=(0, 0, 255), thickness=2)
    cv2.polylines(bgr, [curve], isClosed=False, color=(0, 0, 255), thickness=2)
    cv2.imshow("Result", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    paths.append(curve)

    # for contour in contours:
    #     print("before length", len(contour))
    #     points = fit_curve(contour, len(contour)/2)
    #     print("after length", len(points))

    #paths = extract_single_contour_from_contours(contours, bgr)
    
    # draw each line in different color and thickness 2
    # 8 colors
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]

    # draw the lines in different colors above
    for i, path in enumerate(paths):
        print("drawing line with path len", len(path))
        for j in range(len(path) - 1):            
            cv2.line(bgr, tuple(path[j]), tuple(path[j+1]), colors[i % len(colors)], 2)

        # draw a circle at the start and end of the path
        cv2.circle(bgr, tuple(path[0]), 5, (0, 255, 255), -1)
        cv2.circle(bgr, tuple(path[-1]), 5, (0, 255, 255), -1)

    # # plot the paths in the same image
    # for path in paths:
    #     for i in range(len(path) - 1):
    #         cv2.line(bgr, tuple(path[i]), tuple(path[i+1]), (0, 0, 255), 2)

    cv2.imshow("paths", bgr)
    cv2.waitKey(0)

    def get_distance(path1, path2):
        return np.linalg.norm(path1 - path2)
    
    def bezier_curve(t, points):
        """Calculate a point on a Bézier curve defined by control points."""
        #print("control from bezier_curve()", t)
        n = len(points) - 1
        x, y = 0, 0
        for i, (px, py) in enumerate(points):
            coeff = math.comb(n, i) * (1 - t)**(n - i) * t**i
            x += coeff * px
            y += coeff * py
        return x, y

    def generate_high_res_points(source, target, control_points, resolution=1):
        """Generate high-resolution points along the Bézier curve."""
        points = [source] + control_points + [target]
        curve_length = sum(np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1))
        num_points = int(curve_length / resolution)

        t_values = np.linspace(0, 1, num_points)
        high_res_points = [bezier_curve(t, points) for t in t_values]
        return np.array(high_res_points)
    
    # high_resolution_paths = []
    # for path in paths:
    #     print("before type", type(path))
    #     print("before *********path length", len(path))
    #     path = generate_high_res_points(path[0], path[-1], path[1:-1], resolution=0.5)
    #     print("after type", type(path))
    #     print("before ^^^^^^^^^^^path length", len(path))
    #     if len(path) > 0:
    #         high_resolution_paths.append(path)
    
    # paths = high_resolution_paths

    # # Sort the paths based on the distance between the end of one path and the start of the next
    # do this for all paths, they can be in any order
    print("total paths", len(paths))

    # write all paths to a file (dynamically named)
    f = open("current_points.txt", "w")
    for path in paths:
        f.write("\n === path start ==== \n")

        # write first 5 points and last 5 points
        for i, point in enumerate(path):
            if i < 5 or i > len(path) - 6:
                x, y = point
                f.write(f"[{x}, {y}]\n")
        
    final_path = paths[0]
    paths.pop(0)
    threshold = 15
    consecquent_errors = 0
    while len(paths) > 0:
        for i, path in enumerate(paths):
            # if len(path) <= 10:
            #     paths.pop(i)
            #     continue
            start_to_end = get_distance(final_path[0], path[-1])
            end_to_start = get_distance(final_path[-1], path[0])
            start_to_start = get_distance(final_path[0], path[0])
            end_to_end = get_distance(final_path[-1], path[-1])
            print("trying to merge contour", i)
            print("start to end and (start, end)", start_to_end, final_path[0], path[-1])
            print("end to start and (end, start)", end_to_start, final_path[-1], path[0])
            print("start to start and (start, start)", start_to_start, final_path[0], path[0])
            print("end to end and (end, end)", end_to_end, final_path[-1], path[-1])

            if start_to_end < threshold:
                print("start to end", start_to_end)
                final_path = np.concatenate((path, final_path))
                paths.pop(i)
                consecquent_errors = 0
                break
            elif end_to_start < threshold:
                print("end to start", end_to_start)
                final_path = np.concatenate((final_path, path))
                paths.pop(i)
                consecquent_errors = 0
                break
            elif start_to_start < threshold:
                print("start to start", start_to_start)
                final_path = np.concatenate((path[::-1], final_path))
                paths.pop(i)
                consecquent_errors = 0
                break
            elif end_to_end < threshold:
                print("end to end", end_to_end)
                final_path = np.concatenate((final_path, path[::-1]))
                paths.pop(i)
                consecquent_errors = 0
                break
            else:
                print("No match found")
                consecquent_errors += 1

            # see if any of these lines are over another, for example, when the path is reversing
            # there can be chances that one path/line is over another
            # identify such cases and try to merge them
            # if there are no matches, but the path is over another path, then try to merge them
     
            for j in range(len(final_path) - 1):            
                cv2.line(bgr, tuple(final_path[j]), tuple(final_path[j+1]), (255, 255, 255), 2)

            # draw a circle at the start and end of the final path
            cv2.circle(bgr, tuple(final_path[0]), 5, (0, 0, 255), -1)
            cv2.circle(bgr, tuple(final_path[-1]), 5, (0, 0, 255), -1)            

            cv2.imshow("new path - ", bgr)
            cv2.waitKey(0)

            if consecquent_errors > 5:
                print("*****************Could not proceed to merge")
                exit(1)

        if consecquent_errors > 5:
                print("###############Could not proceed to merge")
                exit(1)

        

    
    # Print the ordered coordinates
    # f = open("tmp.txt", "w")
    # for i, point in enumerate(final_path):
    #     # print("iter", i)
    #     x, y = point
    #     f.write(f"[{x}, {y}]\n")
    # f.close()
    return final_path