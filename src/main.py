from FeatureExtractor import FeatureExtractor
from Utils import utils
import numpy as np
import cv2
import timeit


from matplotlib import pyplot as plt

img = cv2.imread("../data/1.png", 0)
# Reduce the image size
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Original Image', img)
feature_extractor = FeatureExtractor.FeatureExtractor(5)

start = timeit.default_timer()
srs_masks = feature_extractor.get_srs_images(img)
stop = timeit.default_timer()

print('Time: ', stop - start)
utils.show_images(srs_masks)

cv2.destroyAllWindows()




