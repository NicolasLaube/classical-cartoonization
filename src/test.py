import os
import sys

import cv2 as cv

sys.path.append(os.path.join("..", ".."))

img1 = cv.imread("data/frame0-36-19.35.jpg")
img2 = cv.imread("data/original.jpg")
print(img2)

# Images have to be of the same size to be added
# so resize one image to the size of the other before adding
img2 = cv.resize(img2, (256, 256))
img1 = cv.resize(img1, (256, 256))

# add(or blend) the images
result = cv.addWeighted(img1, 0.3, img2, 0.7, 0)
# create a resizable window for the image
cv.namedWindow("result", cv.WINDOW_NORMAL)
# show the image on the screen
cv.imshow("result", result)
if cv.waitKey() == 0:
    cv.destroyAllWindows()

# add(or blend) the images
result = cv.addWeighted(img1, 0.3, img2, 0.7, 0)
# create a resizable window for the image
cv.namedWindow("result", cv.WINDOW_NORMAL)
# show the image on the screen
cv.imshow("result", result)
if cv.waitKey() == 0:
    cv.destroyAllWindows()
