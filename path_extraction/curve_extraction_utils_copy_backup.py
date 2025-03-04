import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

     
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
    for contour in contours:
        # Reshape the contour to a list of (x, y) points
        contour_points = contour.reshape(-1, 2)
        #print("contour_points", contour_points)

        # If the contour is a loop, break the loop to form a line-like structure
        # We assume the endpoints are the points with maximum distance from each other
        distances = np.linalg.norm(contour_points - contour_points[:, None], axis=2)
        start_idx, end_idx = np.unravel_index(distances.argmax(), distances.shape)
        print("start_idx, end_idx", start_idx, end_idx)
        if start_idx > end_idx:
            contour_points = contour_points[start_idx:end_idx:-1]
        else:
            contour_points = contour_points[start_idx : end_idx + 1]

        paths.append(contour_points)    
    
    # draw each line in different color and thickness 2
    # 8 colors
    colors = [(100, 200, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]

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
                threshold += 10

            if consecquent_errors > 15:
                print("consecquent_errors", consecquent_errors)
                print("Could not find a match for the path")
                exit(1)


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

        if consecquent_errors == 0:
            threshold = 15
                

        

    
    # Print the ordered coordinates
    # f = open("tmp.txt", "w")
    # for i, point in enumerate(final_path):
    #     # print("iter", i)
    #     x, y = point
    #     f.write(f"[{x}, {y}]\n")
    # f.close()
    return final_path