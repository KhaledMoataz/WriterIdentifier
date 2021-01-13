from src.FeatureExtractor import FeatureExtractor
import numpy as np
import cv2
import timeit


from matplotlib import pyplot as plt

img = cv2.imread("../data/1.png", 0)
# Reduce the image size
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# img = np.zeros((3, 3),  dtype=np.uint8)
# for i in range(3):
#     for j in range(3):
#         img[i][j] = 3 * i + j
# print(img)
# plt.hist(img.ravel(),256,[0,256]); plt.show()
# ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# print(ret)
# cv2.imshow('win1', thr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imshow('Original Image', img)
feature_extractor = FeatureExtractor.FeatureExtractor(2)

start = timeit.default_timer()


normal_mask, mask = feature_extractor.getLBP(img, 1)

stop = timeit.default_timer()

print('Time: ', stop - start)

cv2.imshow('Normal Mask', normal_mask)
cv2.imshow('Modified Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()




