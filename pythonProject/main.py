import cv2

print("21811954")
print("박원영")

image = cv2.imread("./testImage.png")
cv2.imshow("test image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()