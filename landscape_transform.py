#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  landscape_transform.py [--db=<path to image directory>] [<image1> <image2>]

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import sys, getopt

from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.xfeatures2d.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(200)
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

def crop_img(dir_name, img_name, height_crop=30, width_crop=0):
    cropped_dir_name = dir_name + '_cropped'
    if not os.path.exists(cropped_dir_name):
        os.makedirs(cropped_dir_name)

    img = cv.imread(dir_name + '/' + img_name)
    y = img.shape[0]
    x = img.shape[1]

    h=height_crop
    w=width_crop
    crop_img = img[h:y-h, w:x-w]

    # cv.imshow("cropped", crop_img)
    # cv.waitKey()
    cv.imwrite(cropped_dir_name + '/cropped-' + img_name, crop_img)

def compare_imgs(fn1, fn2, features):
    img1 = cv.imread(fn1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(fn2, cv.IMREAD_GRAYSCALE)

    for f in features:
        # detector, matcher = init_feature(feature_name)
        detector, matcher = init_feature(f)

        if img1 is None:
            print('Failed to load fn1:', fn1)
            sys.exit(1)

        if img2 is None:
            print('Failed to load fn2:', fn2)
            sys.exit(1)

        if detector is None:
            print('unknown feature:', f)
            sys.exit(1)

        print('using', f)

        kp1, desc1 = detector.detectAndCompute(img1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

        def match_and_draw(win):
            print('matching...')
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, 0.3)
            if len(p1) >= 4:
                H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
                print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            else:
                H, status = None, None
                print('%d matches found, not enough for homography estimation' % len(p1))

            _vis = explore_match(win, img1, img2, kp_pairs, status, H)

        match_and_draw(f)
        cv.waitKey()
        cv.destroyAllWindows()

    print('Done')

def main():
    opts, args = getopt.getopt(sys.argv[1:], '', ['db='])
    opts = dict(opts)
    db_directory = opts.get('--db', 'landscape_db')
    try:
        fn1, fn2 = args
    except:
        fn1 = 'imgs/ref1.jpg'
        fn2 = 'imgs/test1.jpg'

    # features = ["sift", "surf", "orb", "akaze", "brisk"]
    # features = ["sift", "surf", "orb"]
    features = ["surf"]
    # compare_imgs(fn1, fn2, features)

    # crop images
    images = os.listdir(db_directory)
    for filename in images:
        # print(filename)
        crop_img(db_directory, filename)

    features = ["surf"]
    images = os.listdir(db_directory + "_cropped")
    for i in range(0, len(images) - 1):
        prefix = db_directory + '_cropped/'
        db_img_1 = prefix + images[i]
        db_img_2 = prefix + images[i+1]
        compare_imgs(db_img_1, db_img_2, features)

        # img_temp = cv.imread(db_directory + '/' + filename, cv.IMREAD_GRAYSCALE)
        # kp_temp, desc_temp = detector.detectAndCompute(img_temp, None)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
