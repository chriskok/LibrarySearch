import cv2
img = cv2.imread("imgs/test1.jpg")
y = img.shape[0]
x = img.shape[1]

h=30
w=0
crop_img = img[h:y-h, 0:x]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
