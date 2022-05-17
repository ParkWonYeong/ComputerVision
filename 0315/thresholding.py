import cv2

image = cv2.imread("./test3.png")

Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Gray)

# Global
ret, dst = cv2.threshold(Gray, 127,255, cv2.THRESH_BINARY)

# Adaptive
# dst = cv2.adaptiveThreshold(Gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                             cv2.THRESH_BINARY, 15, 4)

cv2.imshow("Binarized image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()