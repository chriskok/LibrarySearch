#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
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


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv.circle(vis, (x1, y1), 2, col, -1)
            cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]

            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i][0]]
                cv.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv.imshow(win, cur_vis)
    cv.setMouseCallback(win, onmouse)
    return vis


def main():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk')
    try:
        fn1, fn2 = args
    except:
        fn1 = 'box.png'
        fn2 = 'box_in_scene.png'

    # initialize the camera
    cap = cv.VideoCapture(0)   # 0 -> index of camera

    detector, matcher = init_feature(feature_name)

    print('using', feature_name)

    kp_list = []
    desc_list = []
    img_list = []
    for filename in os.listdir('test_imgs'):
        img_temp = cv.imread('test_imgs/' + filename, cv.IMREAD_GRAYSCALE)
        kp_temp, desc_temp = detector.detectAndCompute(img_temp, None)
        img_list.append(img_temp)
        kp_list.append(kp_temp)
        desc_list.append(desc_temp)

    # print(str(len(kp_list)) + ", " + str(len(desc_list)))

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
            desc2 = np.float32(desc2)
            desc_list[i] = np.float32(desc_list[i])

            matches = flann.knnMatch(desc_list[i],desc2,k=2)

            # if (len(matches) > best_match_count):
            #     best_match = matches
            #     best_match_count = len(matches)
            #     best_match_index = matches_index
            #

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
