# import required libraries
import cv2
import numpy as np

# read input image
img = cv2.imread("test2.jpg")

# Convert BGR to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # define range of blue color in HSV
# lower_yellow = np.array([15, 50, 180])
# upper_yellow = np.array([40, 255, 255])

# # Create a mask. Threshold the HSV image to get only yellow colors
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# lower mask (0-10)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0 + mask1

# Bitwise-AND mask and original image
result = cv2.bitwise_and(img, img, mask=mask)

# display the mask and masked image
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.imshow("Masked Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
