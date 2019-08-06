from pyflann import *
import numpy as np

dataset = np.array(
    [[[1., 1, 1, 2, 3], [1., 1, 1, 2, 3]],
     [[10, 10, 10, 3, 2], [1., 1, 1, 2, 3]],
     [[100, 100, 2, 30, 1], [1., 1, 1, 2, 3]]
     ])
testset = np.array(
    [[[1., 1, 1, 1, 1], [1., 1, 1, 2, 3]],
     [[90, 90, 10, 10, 1], [1., 1, 1, 2, 3]]
     ])
# flann = FLANN()
# result, dists = flann.nn(
#     dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
# print result
# print dists

# Find the nearest 5 neighbors of point 100.
flann = FLANN()
neighbors = flann.nn(dataset, testset[1,], num_neighbors = 3)
print("Nearest neighbors of point 100: ")
print(neighbors[0])
print("Distances: ")
print(neighbors[1])

# dataset = np.random.rand(10000, 128)
# testset = np.random.rand(1000, 128)
# flann = FLANN()
# result, dists = flann.nn(
#     dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
# print result
# print dists
