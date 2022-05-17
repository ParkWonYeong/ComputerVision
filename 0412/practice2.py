### Practice1 ###

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('book.jpg')
img2 = cv2.imread('image.jpg')
# cv2.imshow("Query image", img1)
# cv2.imshow("Reference image", img2)

# create SIFT feature extractor object
sift = cv2.SIFT_create()

# find the keypoints, descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# brute force matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)      # matching할 대상(이미지) 개수가 들어가는 듯하다. 세 장의 이미지일 경우 k=3이 될 것으로 예상.

# ratio distance
good = []
for m, n in matches:
    if m.distance < 0.5*n.distance:     # best match에 대한 ratio distance의 임계값인 듯하다.
        good.append([m])

img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=6)    # flags는 Blob이 보이는 채널??같은것같다. 2의 n제곱 값이 할당되지 않으면 에러 발생.

cv2.imshow("matching", img4), plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()