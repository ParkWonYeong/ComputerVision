##Practice1##

import cv2
import numpy as np

image = cv2.imread("./test_image.jpg")
cv2.imshow("Image", image)

src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# kernel1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])    # 필터1
kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])    # 필터2
# 제안하는 kernel filter
kernel = np.array([[1, 2, -1], [1, 0, -1], [1, -2, -1]])
kernel4 = np.array([[1, 2, 1], [0, 0, 0], [1, -2, -1]])
# dst1 = cv2.filter2D(src, -1, kernel1)   # 필터1 결과
dst2 = cv2.filter2D(src, -1, kernel2)   # 필터2 결과
dst = cv2.filter2D(src, -1, kernel)
dst4 = cv2.filter2D(src, -1, kernel4)

# cv2.imshow("Filtered Image(1)", dst1)
cv2.imshow("Filtered Image(2)", dst2)
cv2.imshow("Filtered Image(3)", dst)
cv2.imshow("Filtered Image(4)", dst4)

cv2.waitKey(0)
cv2.destroyAllWindows()