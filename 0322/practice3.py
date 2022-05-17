##Practice3##
import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)
print(kernel)

path = "./test_img3.jpg"
img = cv2.imread(path)
# cv2.imshow("image", img)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("GrayScale", imgGray)

imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)      # Kernel size = (3,3), sigma = 0
# cv2.imshow("Img Blur", imgBlur)

imgCanny = cv2.Canny(imgBlur, 30, 120)              # th1 = 30, th2 = 120
imgCanny2 = cv2.Canny(imgBlur, 40, 120)

cv2.imshow("Img Canny", imgCanny)
cv2.imshow("Img Canny1", imgCanny2)

cv2.waitKey(0)
cv2.destroyAllWindows()

## 3.3.4 ##
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("./test_img3.jpg")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
# # imgCanny = cv2.Canny(imgBlur,100,200)
#
# fig = plt.figure(figsize=(20, 10))
# fig.canvas.manager.set_window_title('The Effects of the Lower Threshold Value on Canny')
#
# for i, value in enumerate(range(10, 181, 10)):
#     print("i, value : ", i, value)
#     canny = cv2.Canny(imgBlur, value, 200)
#     plt.subplot(3, 6, i+1), plt.title('Lower Threshold = {}'.format(value))
#     plt.imshow(canny, cmap='gray')
#
# fig = plt.figure(figsize=(20, 10))
# fig.canvas.manager.set_window_title('The Effects of the Higher Threshold Value on Canny')
# for j, value2 in enumerate(range(20, 251, 10)):
#     print("j, value : ", j, value2)
#     canny2 = cv2.Canny(imgBlur, 100, value2)
#     plt.subplot(4, 6, j+1), plt.title('Higher Threshold = {}'.format(value2))
#     plt.imshow(canny2, cmap='gray')
#
# plt.tight_layout()
# plt.show()