import numpy as np
import cv2

def compute_cumulative_distance(points):
    """Compute cumulative traversal distance along the looped curve."""
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return distances

def find_farthest_points_on_loop(points):
    """
    Finds the two farthest points on a closed curve using accumulated traversal distance.

    :param points: np.array of shape (N, 2), where each row is (x, y)
    :return: (index1, index2) of the two farthest points
    """
    points = np.array(points)  # Ensure it's a NumPy array

    # Step 1: Compute cumulative traversal distance along the curve
    distances = compute_cumulative_distance(points)

    # Step 2: Find total curve length
    total_length = distances[-1]

    # Step 3: Find the point where distance reaches half of total length
    half_distance = total_length / 2
    idx2 = np.searchsorted(distances, half_distance)  # Find the closest index

    # Start point is always the first point
    idx1 = 0  

    return idx1, idx2, total_length
def extract_curve_from_mask_v1(image: cv2.typing.MatLike):
    # Convert the image to grayscale
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 1)

    

    # Ensure the image is binary
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours in the skeletonized image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(binary_image, contours, -1, (255, 255, 255), 1)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    

    #draw contours with color blue and thickness 5
    
    # cv2.imshow("contours", bgr)
    # cv2.waitKey(0)

    paths = []
    colors = [(100, 200, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (100, 105, 255), (0, 0, 0)]
    for i, contour in enumerate(contours):
        # Reshape the contour to a list of (x, y) points
        contour_points = contour.reshape(-1, 2)        

        idx1, idx2, total_length = find_farthest_points_on_loop(contour_points)

        if (total_length < 5):
            continue

        print(f"Farthest points indices: {idx1}, {idx2}")
        print(f"Farthest points: {contour_points[idx1]}, {contour_points[idx2]}")        
        print("total length of the contour", total_length)
        print(f"first point {contour_points[0]}, last point {contour_points[-1]}")        


        # copy all the points from start to end
        path = contour_points[idx1:idx2 + 1]

        # # Plot the looped curve and mark the farthest points
        # plt.plot(path[:, 0], path[:, 1], 'bo-', label="Looped Curve")
        # plt.scatter([path[idx1, 0], path[idx2, 0]],
        #             [path[idx1, 1], path[idx2, 1]],
        #             color='red', s=100, label="Farthest Points")
        # plt.legend()
        # plt.title("Farthest Points on a Closed Curve")
        # plt.show()

        print("drawing line with path len", len(path))
        cv2.drawContours(bgr, [contour], -1, (255, 255, 255), 2)
        for j in range(len(path) - 1):            
            cv2.line(bgr, tuple(path[j]), tuple(path[j+1]), colors[i % len(colors)], 2)
        cv2.circle(bgr, tuple(path[0]), 5, (0, 0, 255), -1)
        cv2.circle(bgr, tuple(path[-1]), 5, (0, 0, 255), -1)

        cv2.circle(bgr, tuple(contour_points[idx1]), 5, (255, 200, 0), -1)
        cv2.circle(bgr, tuple(contour_points[idx2]), 5, (255, 200, 0), -1)

        cv2.imshow("paths", bgr)
        cv2.waitKey(0)
        
        paths.append(path) 

    # cv2.imshow("final image", bgr)
    # cv2.waitKey(0)
    return paths
        


# if __name__ == "__main__":
    
#     image = cv2.imread("../data/mask_3.png", cv2.IMREAD_COLOR)
#     extract_curve_from_mask(image)
