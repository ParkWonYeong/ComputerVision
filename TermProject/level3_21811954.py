## level 3 : 무작위 9단계 Stitchiing ##

from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import deque

global check
check = 0

def main_code(img1, img2, orimg):

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

        matchesMask = mask.ravel().tolist()

        h, w, ch = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are fount - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    ########## Image Matching ##########

    draw_params = dict(matchColor = (0, 255, 0),        #green
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'binary'), plt.show()

    # img1: bottom view, img2: top view


    global check

    if check == 0:
        height = min(img1.shape[0], img2.shape[0])
        width = img1.shape[1]+img2.shape[1]
    elif check == 1:
        height = img1.shape[0]+img2.shape[0]
        width = max(img1.shape[1], img2.shape[1])


    dst = cv2.warpPerspective(orimg, M, (width, height))

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imshow("Stitching", dst), plt.show()

    return dst

## Call 9 Images from File
Image = []
for _ in range(9):
    root = Tk()
    path = filedialog.askopenfilename(initialdir = "D:/DATA_", title = "choose first image", filetypes = (("jpeg files", "*jpg"), ("all files", "*.*")))
    Image.append(cv2.imread(path))
    root.withdraw()

### count matching points ###
count = []
for i in range(9):
    Img = Image
    a = Img[i]
    print(len(Img))
    cnt = 0
    for j in range(9):
        if i == j:
            pass
        sift = cv2.SIFT_create()
        print(i, j)
        kp1, des1 = sift.detectAndCompute(a, None)
        kp2, des2 = sift.detectAndCompute(Img[j], None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        cnt += len(good)
    count.append(cnt)

# Image, count 배열을 zip을 사용하여 하나의 dictionary로 묶는다.
dictionary = dict(zip(count, Image))    # {count(key):Image(value)}
# Key 값(count 값) 기준 내림차 순으로 정렬하고 1차원 배열로 변환
dictionary = dict(sorted(dictionary.items(), key=lambda x: x[0], reverse=True))
dictionary = list(dictionary.values())


########## check (2,1), (2,3) ##########
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(dictionary[1], None)
kp2, des2 = sift.detectAndCompute(dictionary[2], None)
# FLANN_INDEX_KDTREE = 0 = algorithm
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good.append(m)
if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M[1][2] < 0:  # 반대로 들어온 값이라서 음수인 경우
        print("반대 정렬")
        temp = dictionary[1]
        dictionary[1] = dictionary[2]
        dictionary[2] = temp
    else:
        print("-")


########## check (1,2), (3,2) ##########
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(dictionary[3], None)
kp2, des2 = sift.detectAndCompute(dictionary[4], None)
# FLANN_INDEX_KDTREE = 0 = algorithm
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good.append(m)
if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M[1][2] < 0:  # 반대로 들어온 값이라서 음수인 경우
        print("반대 정렬")
        temp = dictionary[3]
        dictionary[3] = dictionary[4]
        dictionary[4] = temp
    else:
        print("-")

########## check (1,1), (1,3), (3,1), (3,3) ##########
count_b = []
for i in range(4):
    check_4 = dictionary[5:9]
    b = check_4[i]
    cnt_b = 0
    for j in range(4):
        if i==j:
            pass
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(b, None)
        kp2, des2 = sift.detectAndCompute(check_4[j], None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        if len(good) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M[1][2] < 0:  # 반대로 들어온 값이라서 음수인 경우(상대적으로 왼쪽 위의 영상인 경우)
                print("반대 정렬")
                cnt_b += 1      # 반대로 들어왔을 경우의 패널티(cnt_b)를 +1 한다.
                # 패널티가 가장 많이 쌓인 영상이 (1,1)이 된다.
            else:
                print("-")

    # 중복방지(dict사용)
    if cnt_b in count_b:
        cnt_b -= 1
        if cnt_b in count_b:
            cnt_b -= 1
            if cnt_b in count_b:
                cnt_b -= 1
                count_b.append(cnt_b)
        else:
            count_b.append(cnt_b)
    else:
        count_b.append(cnt_b)

# check_4, count_b 배열을 zip을 사용하여 하나의 dictionary로 묶는다.
check_4 = dictionary[5:9]
dict_4 = dict(zip(count_b, check_4))    # {count(key):Image(value)}

# Key 값(count 값) 기준 내림차 순으로 정렬
dict_4 = dict(sorted(dict_4.items(), key=lambda x: x[0], reverse=True))
dict_4 = list(dict_4.values())

dictionary[5] = dict_4[0]    # (1,1)
dictionary[6] = dict_4[1]    # (1,3) 혹은 (3,1)
dictionary[7] = dict_4[2]    # (3,1) 혹은 (1,3)
dictionary[8] = dict_4[3]    # (3,3)

cv2.imshow("(1,1)", dictionary[6]), plt.show()
cv2.imshow("(1,2)", dictionary[2]), plt.show()
cv2.imshow("(1,3)", dictionary[5]), plt.show()
cv2.imshow("(2,1)", dictionary[4]), plt.show()
cv2.imshow("(2,2)", dictionary[0]), plt.show()
cv2.imshow("(2,3)", dictionary[1]), plt.show()
cv2.imshow("(3,1)", dictionary[8]), plt.show()
cv2.imshow("(3,2)", dictionary[3]), plt.show()
cv2.imshow("(3,3)", dictionary[7]), plt.show()

# Last Stitching
first_1 = main_code(dictionary[5], dictionary[2], dictionary[5])
second_1 = main_code(first_1, dictionary[6], first_1)

first_2 = main_code(dictionary[1], dictionary[0], dictionary[1])
second_2 = main_code(first_2, dictionary[4], first_2)

first_3 = main_code(dictionary[7], dictionary[3], dictionary[7])
second_3 = main_code(first_3, dictionary[8], first_3)

check = 1
stitch1 = main_code(second_3, second_2, second_3)
print('stitched 3&2 rows')
stitch2 = main_code(stitch1, second_1, stitch1)
print('stitched 3&2&1 rows')

cv2.waitKey(0)
cv2.destroyAllWindows()