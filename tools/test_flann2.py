import cv2
import numpy as np
import os
from pyflann import *

surf = cv2.xfeatures2d.SURF_create(1000)
flann = cv2.flann_Index()
des_all = np.empty((1, 64), dtype=np.float32)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# foldername = 'downloads/oreilly'
foldername = 'test_imgs'
for index, filename in enumerate(os.listdir(foldername)):
    img2 = cv2.imread(foldername + '/' + filename, 0)
    kp2, des2 = surf.detectAndCompute(img2, None)
    print(des2.shape)

    des_all = np.concatenate((des_all, des2))

img1 = cv2.imread('imgs/test_book_capture.jpg', cv2.IMREAD_GRAYSCALE)
kp2, desc2 = surf.detectAndCompute(img1, None)

# Find the nearest 5 neighbors of point 100.
flann2 = FLANN()
neighbors = flann2.nn(des_all, desc2, num_neighbors = 5)
print("Nearest neighbors of point 100: ")
print(neighbors[0])
print("Distances: ")
print(neighbors[1])

print(des_all.shape)
des_all = np.float32(des_all)

print "Training..."
flann.build(des_all, index_params)

print(flann.getDistance())
print(flann.getAlgorithm())

print(flann.knnSearch(desc2, 2))
