# practice 1

import cv2
import numpy as np

image = cv2.imread("./test_image.jpg")
Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Gray)

# 1-(1)
# kernel = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1.5]])
# dst = cv2.filter2D(Gray, -1, kernel)

# 1-(2)
kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - 1/9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
dst = cv2.filter2D(Gray, -1, kernel)

cv2.imshow("filtered image_Practice 1-(2)", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()