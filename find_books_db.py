import cv2
import numpy as np
import os

# cap = cv2.VideoCapture(1)
# ret, frame = cap.read()
# cv2.imwrite('imgs/test_book_capture.jpg', frame)

scanned = 'imgs/test_book_capture2.jpg'
surf = cv2.xfeatures2d.SURF_create(400)
surf.setUpright(True)

img1 = cv2.imread(scanned, cv2.IMREAD_GRAYSCALE)
kp1, des1 = surf.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_MATCH_COUNT = 10
def flann_match(des1):
    matches = flann.knnMatch(des1, k=2)
    # Check the distance to keep only the good matches.
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
    if len(matches) < MIN_MATCH_COUNT:
        return False
    return True

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


for filename in os.listdir('test_imgs'):
    img2 = cv2.imread('test_imgs/' + filename, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = surf.detectAndCompute(img2, None)

    # matches = flann.knnMatch(des2, des1, k=2)

    # Match descriptors.
    matches = bf.match(des2,des1)
    good_points = matches

    # good_points = []
    # for m, n in matches:
    #     if m.distance > 0.8*n.distance:
    #         good_points.append(m)
    number_keypoints = 0
    if len(kp1) >= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)
    print("Title: " + filename)
    print(len(matches))
    print("keypoints: " + str(number_keypoints) + ", good_points: " + str(len(good_points)))
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    print("Similarity: " + str(percentage_similarity) + "\n")


# # initialize the camera
# cap = cv.VideoCapture(1)   # 0 -> index of camera
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
