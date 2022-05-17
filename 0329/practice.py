## practice : 22-03-29

import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('./meow1.jpg')
# img = cv2.resize(img, (600, 800))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img2 = cv2.imread('./meow2.jpg')
# img2 = cv2.resize(img2, (600, 800))
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img = cv2.imread('./test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
# kernel = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
# dst = cv2.filter2D(gray, -1, kernel)
# dst2 = cv2.filter2D(gray2, -1, kernel)

# Canny
# imgBlur = cv2.GaussianBlur(gray, (3,3), 5)
# imgBlur2 = cv2.GaussianBlur(gray2, (3,3), 5)
# imgCanny = cv2.Canny(imgBlur, 30, 120)
# imgCanny = cv2.GaussianBlur(imgCanny, (5,5), 2)
# imgCanny2 = cv2.Canny(imgBlur2, 30, 120)
# imgCanny2 = cv2.GaussianBlur(imgCanny2, (5,5), 2)
# cv2.imshow("Img Canny", imgCanny)
# cv2.imshow("Img Canny2", imgCanny2)

##### #2, #3 추가 #####
# Gaussian Blur
# imgBlur = cv2.GaussianBlur(imgCanny, (3, 3), 0)      # Kernel size = (3,3), sigma = 0
# imgBlur2 = cv2.GaussianBlur(imgCanny2, (3, 3), 0)      # Kernel size = (3,3), sigma = 0

##### #2, #3 추가 #####

# Sharpening(1)
# alpha = 1
# imgSharp = gray + alpha*(gray-imgBlur)
# imgSharp2 = gray2 + alpha*(gray2-imgBlur2)

# Sharpening(2)
kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - 0.8*1/9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
imgSharp = cv2.filter2D(gray, -1, kernel)
# imgSharp2 = cv2.filter2D(gray2, -1, kernel)
##### #2, #3 추가 #####

cv2.imshow('imgSharp', imgSharp)

imgSharp = np.float32(imgSharp)
dst = cv2.cornerHarris(imgSharp, 2, 3, 0.04)    # Use Harris Corner Detector

# imgBlur2 = np.float32(imgBlur2)
# dst2 = cv2.cornerHarris(imgBlur2, 2, 3, 0.04)    # Use Harris Corner Detector
# img - Input Image
# blockSize - the size of neighborhood considered for corner detection
# ksize - Aperture parameter of Sobel derivative used
# k - Harris detector free parameter in the equation
# + 픽셀보간법





# 검출된 코너의 부분을 좀더 확대하기 위해서 dilate를 적용한다.
dst = cv2.dilate(dst, None)
plt.figure()
plt.imshow(dst)
plt.colorbar()

# dst2 = cv2.dilate(dst2, None)
# plt.figure()
# plt.imshow(dst2)
# plt.colorbar()

# Threshold for an optimal value: depending on the image
img[dst > 0.06*dst.max()] = [0, 255, 0]
# img2[dst2 > 0.01*dst2.max()] = [0, 0, 255]

cv2.imshow('dst_img', img)
# cv2.imshow('dst2_img', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()