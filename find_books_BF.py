#!/usr/bin/env python
'''
Feature-based image matching accross a database.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SURF

USAGE
  find_books.py [--cam=<0|1>]

  --cam - camera index to use if you have multiple webcams available

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from PIL import Image
import os

from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.xfeatures2d.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(1000)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher

def main():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['cam=', 'db='])
    opts = dict(opts)
    cam_number = opts.get('--cam', 0)
    db_directory = opts.get('--db', 'test_imgs')

    # initialize the camera
    cap = cv.VideoCapture(int(cam_number))

    detector, matcher = init_feature('surf')

    kp_list = []
    desc_list = []
    img_list = []
    for filename in os.listdir(db_directory):
        img_temp = cv.imread(db_directory + '/' + filename, cv.IMREAD_GRAYSCALE)
        print(filename)
        kp_temp, desc_temp = detector.detectAndCompute(img_temp, None)
        img_list.append(img_temp)
        kp_list.append(kp_temp)
        desc_list.append(desc_temp)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        img2 = gray

        kp2, desc2 = detector.detectAndCompute(img2, None)

        best_match = []
        best_match_percentage = 0
        best_match_index = 0
        matches_index = 0
        for i in range(len(desc_list)):

            matches = flann.knnMatch(desc_list[i],desc2,k=2)

            good_points = []
            for m, n in matches:
                if m.distance < 0.6*n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(kp_list[i]) >= len(kp2):
                number_keypoints = len(kp_list[i])
            else:
                number_keypoints = len(kp2)
            # print("keypoints: " + str(number_keypoints) + ", good_points: " + str(len(good_points)))
            percentage_similarity = float(len(good_points)) / number_keypoints * 100
            # print("Similarity: " + str(percentage_similarity) + "\n")

            if (percentage_similarity > best_match_percentage):
                best_match = good_points
                best_match_index = matches_index
                best_match_percentage = percentage_similarity

            matches_index += 1

        # print("BEST MATCH INDEX: ", best_match_index)



        MIN_MATCH_COUNT = 10

        if len(best_match)>MIN_MATCH_COUNT:
            kp1 = kp_list[best_match_index]
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in best_match ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in best_match ]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img_list[best_match_index].shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # pts = np.array([pts])
            if M is not None:
                dst = cv.perspectiveTransform(pts,M)
                # dst += (w, 0)  # adding offset

                # Draw bounding box in Red
                img2 = cv.polylines(img2, [np.int32(dst)], True, (0,0,255),3, cv.LINE_AA)
            else:
                print("M was none")
        else:
            print ("Not enough matches are found - %d/%d" % (len(best_match),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

        img3 = cv.drawMatches(img_list[best_match_index],kp_list[best_match_index],img2,kp2,best_match, None,**draw_params)

        cv.imshow("result", img3)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
