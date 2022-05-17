##Practice2##
import cv2
import numpy as np

img = cv2.imread("./test_image.jpg")
# cv2.imshow("Image", img)
Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gx = cv2.Sobel(Gray, cv2.CV_32F, 1,0, ksize=3)
gx2 = np.abs(gx)
gx2 = cv2.normalize(gx2, None, 0,1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

gy = cv2.Sobel(Gray, cv2.CV_32F, 0,1, ksize=3)
gy2 = np.abs(gy)
gy2 = cv2.normalize(gy2, None, 0,1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

mag = cv2.magnitude(gx, gy)
mag = cv2.normalize(mag, None, 0,1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
mag2 = cv2.normalize(mag, None, 0,255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("x", gx2)
cv2.imshow("y", gy2)
cv2.imshow("xy(32F)", mag)
cv2.imshow("xy(8U)", mag2)

cv2.waitKey(0)
cv2.destroyAllWindows()