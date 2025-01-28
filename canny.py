import cv2
import cv2.version
import numpy as np
import matplotlib.pylab as plt

# print(cv2.version.opencv_version)

raw_image = cv2.imread("test2.jpg")

gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
value, threshold = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

edges = cv2.Canny(threshold, 100, 300)

plt.subplot(121)
plt.title("edges")
plt.imshow(edges, cmap="gray")
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.title("threshold image")
plt.imshow(threshold, cmap="gray")
plt.xticks([]), plt.yticks([])

plt.show()
