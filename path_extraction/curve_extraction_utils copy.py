import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import distance_matrix, distance
from scipy.spatial.distance import cdist

from scipy.interpolate import CubicSpline

def extract_single_contour_from_contours(contours, image):
    if not contours:
        print("No contours found")
    else:
        paths = []
        for contour in contours:            
            points = contour.squeeze()  # Convert to (N, 2) array

            # Step 1: Split the closed contour into an open path
            # --------------------------------------------------
            # Find the two farthest points
            max_distance = 0
            i1, i2 = 0, 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist > max_distance:
                        max_distance = dist
                        i1, i2 = i, j

            # Split into two segments
            if i1 < i2:
                path1 = points[i1:i2 + 1]
                path2 = np.concatenate((points[i2:], points[:i1 + 1]))
            else:
                path1 = np.concatenate((points[i1:], points[:i2 + 1]))
                path2 = points[i2:i1 + 1]

            # Select the longer path (assumed to be the arrow shaft)
            def compute_length(path):
                return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))

            one_way_path = path1 if compute_length(path1) > compute_length(path2) else path2

            # Step 2: Fit a parametric spline to the open path
            # ------------------------------------------------
            # Create parameter `t` based on point indices
            t = np.arange(len(one_way_path))

            # Fit splines for x(t) and y(t)
            cs_x = CubicSpline(t, one_way_path[:, 0])  # x-coordinate spline
            cs_y = CubicSpline(t, one_way_path[:, 1])  # y-coordinate spline

            # Generate points along the spline (for smoothness)
            t_fine = np.linspace(0, len(one_way_path)-1, 500)  # Adjust density as needed
            x_fit = cs_x(t_fine)
            y_fit = cs_y(t_fine)

            # Combine into (N, 2) array of integers (optional)
            fitted_curve = np.column_stack((x_fit, y_fit)).astype(np.int32)

            # find the length/total distance of the fitted curve and if it is less than 10, ignore it

            # Step 3: Visualize the result
            # ----------------------------
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Draw the original open path (for comparison)
            cv2.polylines(result, [one_way_path.reshape(-1,1,2).astype(np.int32)], 
                        isClosed=False, color=(0, 255, 0), thickness=2)
            
            # Draw the fitted curve (red)
            for pt in fitted_curve:
                cv2.circle(result, tuple(pt), 1, (0, 0, 255), -1)

            cv2.imshow('Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # The final coordinates are in `fitted_curve` (or `one_way_path` for the raw open path)


            paths.append(fitted_curve)
        return paths

        

def test_ccggppt(contours):    
    if not contours:
        print("No contours found")
        return None, None, None

    # Assume the largest contour is the arrow
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()  # Remove unnecessary dimensions

    # Find two farthest points (likely start & end)
    max_dist = 0
    start, end = None, None
    for i in range(len(contour)):
        for j in range(i+1, len(contour)):
            dist = np.linalg.norm(contour[i] - contour[j])
            if dist > max_dist:
                max_dist = dist
                start, end = contour[i], contour[j]

    # Reorder the contour from start to end
    start_idx = np.where((contour == start).all(axis=1))[0][0]
    end_idx = np.where((contour == end).all(axis=1))[0][0]

    if start_idx < end_idx:
        ordered_path = contour[start_idx:end_idx+1]
    else:
        ordered_path = np.vstack([contour[start_idx:], contour[:end_idx+1]])

    return start, end, ordered_path


def fit_curve(xy_points, num_points=100):
    """
    Fit a smooth curve through the given x,y coordinates and return interpolated points.

    :param xy_points: List of (x, y) coordinates.
    :param num_points: Number of points for the interpolated curve.
    :return: Fitted (x, y) coordinates.
    """
    xy_points = np.array(xy_points)
    x, y = xy_points[:, 0], xy_points[:, 1]

    # Fit a B-spline curve
    tck, _ = splprep([x, y], s=1.0)  # s controls smoothness

    # Generate new interpolated points along the fitted curve
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    return np.column_stack((x_new, y_new))    




def split_contours(contours):
    one_way_contours = []
    for contour in contours:
        points = contour.squeeze()  # Convert to (N, 2) array

        # Find the two farthest points
        max_distance = 0
        i1, i2 = 0, 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_distance:
                    max_distance = dist
                    i1, i2 = i, j

        # Split the contour into two paths
        if i1 < i2:
            path1 = points[i1:i2 + 1]
            path2 = np.concatenate((points[i2:], points[:i1 + 1]))
        else:
            path1 = np.concatenate((points[i1:], points[:i2 + 1]))
            path2 = points[i2:i1 + 1]

        # Compute path lengths
        def compute_length(path):
            return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))

        len1 = compute_length(path1)
        len2 = compute_length(path2)

        # Select the longer path
        one_way_contour = path1 if len1 > len2 else path2

        # Reshape for OpenCV compatibility
        one_way_contour = one_way_contour.reshape((-1, 1, 2)).astype(np.int32)
        one_way_contours.append(np.array(one_way_contour))
    return one_way_contours


        # # Optional: Draw the result
        # result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(result, [one_way_contour], -1, (0, 255, 0), 2)
        # cv2.imshow('One-Way Contour', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# def find_farthest_points(contour_points):
#     """Finds the two farthest points in the contour."""
#     pairwise_distances = distance.pdist(contour_points)  # Compute pairwise distances
#     max_dist_idx = np.argmax(pairwise_distances)  # Index of max distance in condensed matrix
#     num_points = len(contour_points)

#     # Convert 1D condensed index to 2D index
#     i = int((2 * num_points - 1 - np.sqrt((2 * num_points - 1)**2 - 8 * max_dist_idx)) / 2)
#     j = max_dist_idx - (i * (2 * num_points - 1 - i)) // 2

#     return contour_points[i], contour_points[j], i, j

def find_farthest_points(contour_points):
    """Finds the two farthest points in the contour."""
    dist_matrix = cdist(contour_points, contour_points, metric="euclidean")
    start_idx, end_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return contour_points[start_idx], contour_points[end_idx], start_idx, end_idx

def reorder_contour(contour_points, start_idx):
    """Reorders the contour to start from the given index and follow the natural path."""
    reordered = np.roll(contour_points, -start_idx, axis=0)
    return reordered

def extract_curve_from_mask(image: cv2.typing.MatLike):
    # Convert the image to grayscale
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours in the skeletonized image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # draw contours with color blue and thickness 5
    cv2.drawContours(bgr, contours, -1, (255, 255, 0), 2)
    cv2.imshow("contours", bgr)
    cv2.waitKey(0)

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
    
    # for contour in contours:
    #     # Reshape the contour to a list of (x, y) points
    #     contour_points = contour.reshape(-1, 2)

    #     # Compute pairwise distances using scipy (fixing the issue)
    #     dist_matrix = distance_matrix(contour_points, contour_points)

    #     # Get the two farthest points
    #     start_idx, end_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    #     # Ensure proper ordering from start to end
    #     if start_idx < end_idx:
    #         ordered_path = contour_points[start_idx:end_idx + 1]
    #     else:
    #         ordered_path = np.vstack([contour_points[start_idx:], contour_points[:end_idx + 1]])

    #     paths.append(ordered_path)

    # for contour in contours:
    #     contour_points = contour.reshape(-1, 2)

    #     if len(contour_points) < 2:
    #         continue

    #     # Find two farthest points
    #     start, end, start_idx, end_idx = find_farthest_points(contour_points)

    #     # Determine shortest path between start and end within contour
    #     if start_idx < end_idx:
    #         ordered_path = contour_points[start_idx:end_idx + 1]
    #     else:
    #         ordered_path = np.concatenate((contour_points[start_idx:], contour_points[:end_idx + 1]))

    #     paths.append(ordered_path)

    # for contour in contours:
    #     contour_points = contour.reshape(-1, 2)

    #     if len(contour_points) < 2:
    #         continue

    #     # Find the two farthest points
    #     start, end, start_idx, end_idx = find_farthest_points(contour_points)

    #     # Reorder contour to start from the start_idx
    #     ordered_contour = reorder_contour(contour_points, start_idx)

    #     paths.append(ordered_contour)


    # for contour in contours:
    #     print("before length", len(contour))
    #     points = fit_curve(contour, len(contour)/2)
    #     print("after length", len(points))

    paths = extract_single_contour_from_contours(contours)
    
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
            for j, path2 in enumerate(paths):


            for j in range(len(final_path) - 1):            
                cv2.line(bgr, tuple(final_path[j]), tuple(final_path[j+1]), (255, 255, 255), 2)

            # draw a circle at the start and end of the final path
            cv2.circle(bgr, tuple(final_path[0]), 5, (0, 0, 255), -1)
            cv2.circle(bgr, tuple(final_path[-1]), 5, (0, 0, 255), -1)            

            cv2.imshow("new path - ", bgr)
            cv2.waitKey(0)

            if consecquent_errors > 10:
                print("*****************Could not proceed to merge")
                exit(1)

        if consecquent_errors > 10:
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