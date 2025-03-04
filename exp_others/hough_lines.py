import cv2
import numpy as np

# Load the image
image = cv2.imread("test3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur and Canny edge detection
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 50, 150)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5
)

# Draw lines and identify coordinates
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Arrows", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
