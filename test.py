from glob import glob

import cv2 as cv

import util

files = sorted(glob("./camvid/*.png"))
refs = sorted(glob("./ref/*.png"))

I = cv.resize(cv.imread(files[20]), (320, 240)) / 255
ref_images = [cv.imread(ref) / 255 for ref in refs]
cv.imshow('', I)
cv.waitKey()
x, y = util.generate_batch(I, ref_images)
cv.imshow('x', cv.resize(x[0], (640, 480)))
cv.waitKey()
cv.imshow('y', cv.resize(y[0], (640, 480)))
cv.waitKey()
cv.imshow('y', cv.resize(y[1], (640, 480)))
cv.waitKey()
cv.imshow('y', cv.resize(y[2], (640, 480)))
cv.waitKey()
cv.imshow('y', cv.resize(y[3], (640, 480)))
cv.waitKey()
# cv.imshow('', cv.flip(I, 0))
# cv.waitKey()
