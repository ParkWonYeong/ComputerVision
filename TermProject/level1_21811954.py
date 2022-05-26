## level 1 : 상하 영상에 대한 사전정보가 없는 두 개 이미지에 대한 stitching ##

import numpy as np
import cv2
from matplotlib import pyplot as plt

global cnt
cnt = 0

def check_order(M, img1, img2):
    global cnt
    if cnt != 0:
        return

    if M[0][2] < 0:                 # 반대로 들어온 값이라서 음수인 경우
        cnt += 1
        print("위: img1")
        print("아래: img2")
        main_code(img2, img1)

    else:
        print("위: img2")
        print("아래: img1")

def main_code(img1, img2):
    MIN_MATCH_COUNT = 4

    ########## Matching ##########

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    # FLANN_INDEX_KDTREE = 0 = algorithm
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO)     # Without RANSAC

        check_order(M, img1, img2)
        orimg = img1
        global cnt
        if cnt == 2:        # 재귀함수로 한번더 호출되는 것을 방지.
            return

        matchesMask = mask.ravel().tolist()

        h, w, ch = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are fount - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    ########## Image Matching ##########

    draw_params = dict(matchColor=(0, 255, 0),  # green
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'binary'), plt.show()

    # img1: Bottom view, img2: Top View
    width = img2.shape[1] + img1.shape[1]
    height = img2.shape[0] + img1.shape[0]

    dst = cv2.warpPerspective(orimg, M, (width, height))

    cv2.imshow("Warping right to left", dst), plt.show()

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2

    cv2.imshow("Stitching", dst), plt.show()

    cnt += 1        # 재귀함수로 한번더 호출되는 것을 방지(반전된 경우 2가 되어 중복 출력을 방지한다.)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('bottom.jpg', 3)
img2 = cv2.imread('top.jpg', 3)
main_code(img1, img2)

