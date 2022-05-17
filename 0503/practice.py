import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./keypad.jpg', 3)     # read as a gray image
# cv2.imshow('Input', img)

rows, cols, ch = img.shape


## case 1: 2D translation ##
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# dst_tr = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('translation', dst_tr)

## case 2: 2D rotation ##
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
# dst_rot = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('rotation', dst_rot)

## case 3: Affine transform ##
# M = np.float32([[1, 0.5, 150], [1, 1, 200]])
# dst = cv2.warpAffine(img, M, (cols, rows))
# plt.subplot(121), plt.imshow(img), plt.title('Input_case3')
# plt.subplot(122), plt.imshow(dst), plt.title('Output_case3')
# plt.show()

## case 4: perspective transform ##
# M = np.float32([[0.3, -0.3, 100], [0, 0.3, 0], [0, -0.002, 1]])
# dst = cv2.warpPerspective(img, M, (400,300))
# plt.subplot(121), plt.imshow(img), plt.title('Input_case4')
# plt.subplot(122), plt.imshow(dst), plt.title('Output_case4')
# plt.show()


# perspective transform
# pts1 = np.float32([[2879, 1425], [3314, 1369], [3242, 1789], [3696, 1710]])   # 왼위, 오위, 왼아래, 오아래
# 반시계 90도 회전이므로 위에서 나타낸 오위, 오아래, 왼위, 왼아래 순으로 나열
pts1 = np.float32([[3314, 1369], [3696, 1710], [2879, 1425], [3242, 1789]])
pts2 = np.float32([[0, 0], [400, 0], [0, 300], [400, 300]])     # 왼위, 오위, 왼아래, 오아래
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (400, 300))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()