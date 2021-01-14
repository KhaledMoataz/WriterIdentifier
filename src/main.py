from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
import numpy as np
import cv2
import timeit


#from matplotlib import pyplot as plt
path = "../data/06"
img1 = utils.read_image(path + "/1/1.png")
img1_ = utils.read_image(path + "/1/2.png")

img2 = utils.read_image(path + "/2/1.png")
img2_ = utils.read_image(path + "/2/2.png")

img3 = utils.read_image(path + "/3/1.png")
img3_ = utils.read_image(path + "/3/2.png")

test_img = utils.read_image(path + "/test.png")


# cv2.imshow('Original Image', img)
feature_extractor = FeatureExtractor.FeatureExtractor(5)
features = feature_extractor.extract_features(img1)
# print(features.shape)
features = np.append(features, feature_extractor.extract_features(img1_), axis=0)
features = np.append(features, feature_extractor.extract_features(img2), axis=0)
features = np.append(features, feature_extractor.extract_features(img2_), axis=0)
features = np.append(features, feature_extractor.extract_features(img3), axis=0)
features = np.append(features, feature_extractor.extract_features(img3_), axis=0)

features = np.append(features, feature_extractor.extract_features(test_img), axis=0)
print(type(features))
print(features.shape)
print(features)
features = feature_extractor.apply_pca(features)
print(type(features))
print(features.shape)
print(features)

classifier = Classifier()
classifier.train(features[:-1], [1, 1, 2, 2, 3, 3])
print(classifier.classify([features[-1]]))
#
# start = timeit.default_timer()
# srs_masks = feature_extractor.get_srs_images(img)
# stop = timeit.default_timer()
#
# print('Time: ', stop - start)
# utils.show_images(srs_masks)
#
# cv2.destroyAllWindows()
#
#
#
#
