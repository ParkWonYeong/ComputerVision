# practice 2
import cv2

image = cv2.imread("./test_image.jpg")
cv2.imshow("Gray", image)

Gaussian1 = cv2.GaussianBlur(image, (3, 3), 0.5)
Gaussian2 = cv2.GaussianBlur(image, (1,1),0.5)
# cv2.imshow("GaussianBlurs", Gaussian)
alpha = 1
image1 = image + alpha*(image - Gaussian1)
image2 = image + alpha*(image - Gaussian2)
cv2.imshow("Gaussian Blur image", image1)
cv2.imshow("Gaussian Blur image2", image2)

cv2.waitKey(0)
cv2.destroyAllWindows()