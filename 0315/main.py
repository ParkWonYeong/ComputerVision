import cv2
import numpy as np

image = cv2.imread("./0315/test_image.jpg")
Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Gray)

kernel = 1/9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
Gray = cv2.filter2D(Gray, -1, kernel)

cv2.imshow("filtered image", Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()