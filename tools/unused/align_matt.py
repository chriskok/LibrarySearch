from __future__ import print_function
import cv2
import numpy as np



def alignImagesORB(im1, im2, MAX_FEATURES, GOOD_MATCH_PERCENT):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  # im1Gray = im1
  # im2Gray = im2
  # cv2.imshow('color?', im2Gray)
  # cv2.waitKey()
  # cv2.destroyAllWindows()

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  # orb = cv2.ORB_create()
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
  matches = bf.match(descriptors1,descriptors2)
  print("key1: {}".format(len(keypoints1)))

# Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)
  # Draw first 10 matches.
  im3 = img = np.zeros((1080,1240,3), np.uint8)
  im3 = cv2.drawMatches(im1Gray,keypoints1,im2Gray,keypoints2,matches[:10],im3)
  cv2.imshow('matches?', im3)
  cv2.waitKey()
  cv2.destroyAllWindows()

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
  print("matches:{}".format(matches))
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  print("matches:{}".format(len(matches)))

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)


  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h

def alignImagesAKAZE(im1, im2, GOOD_MATCH_PERCENT):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.AKAZE_create()
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h

def alignImagesBRISK(im1, im2, GOOD_MATCH_PERCENT):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.BRISK_create()
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h

if __name__ == '__main__':

  # Read reference image
  refFilename = "imgs/test1.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  # imReference = imReference.astype('uint8')
  cv2.imshow('result', imReference)
  cv2.waitKey()
  cv2.destroyAllWindows()

  # Read image to be aligned
  imFilename = "imgs/test1_rotate.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  # im = im.astype('uint8')
  cv2.imshow('result', im)
  cv2.waitKey()
  cv2.destroyAllWindows()

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  MAX_FEATURES = 1000
  GOOD_MATCH_PERCENT = 0.1

  imReg, h = alignImagesORB(imReference, im, MAX_FEATURES, GOOD_MATCH_PERCENT)
  cv2.imshow('result', imReg)
  print("Estimated homography : \n",  h)
  cv2.waitKey()
  cv2.destroyAllWindows()

  for features in range(1, 10):
      MAX_FEATURES = 1000 * features
      for percent in range(1, 10):
          GOOD_MATCH_PERCENT = 0.1 * percent

          imReg, h = alignImagesORB(im, imReference, MAX_FEATURES, GOOD_MATCH_PERCENT)
          cv2.imshow('result', imReg)
          print("Estimated homography : \n",  h)
          cv2.waitKey()
          cv2.destroyAllWindows()

          imReg, h = alignImagesAKAZE(im, imReference, MAX_FEATURES, GOOD_MATCH_PERCENT)
          cv2.imshow('result', imReg)
          print("Estimated homography : \n",  h)
          cv2.waitKey()
          cv2.destroyAllWindows()

          imReg, h = alignImagesBRISK(im, imReference, MAX_FEATURES, GOOD_MATCH_PERCENT)
          cv2.imshow('result', imReg)
          print("Estimated homography : \n",  h)
          cv2.waitKey()
          cv2.destroyAllWindows()
