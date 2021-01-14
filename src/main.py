from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
import numpy as np
import cv2
import timeit
import preprocessor
import random

feature_extractor = FeatureExtractor.FeatureExtractor(5)


def read_ascii(path):
    f = open(path, "r")
    dataset = [[] for _ in range(672)]
    for line in f:
        if line[0] == '#':
            continue
        img_name, writer, a, b, c, d, e, f = line.split()
        dataset[int(writer)].append(img_name)
    return dataset

def extract_features(path):
    # img = utils.read_image(path)
    img = preprocessor.preProcessor(path)
    cv2.imshow("img", img)
    cv2.waitKey(1)
    return feature_extractor.extract_features(img)

dataset = read_ascii("../../ascii/forms.txt")
dataset_length = (len(dataset) // 3) * 3
errors = 0
for i in range(0, dataset_length, 3):
    # writers = random.sample(range(0, len(dataset)), 3)
    writers = [i, i + 1, i + 2]
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for writer in writers:
        pages_count = len(dataset[writer])
        train_images_count = min(2, pages_count)
        train_images += dataset[writer][:train_images_count]
        train_labels += [writer] * train_images_count
        writer_test_paths = dataset[writer][2:]
        test_images += writer_test_paths
        test_labels += [writer] * len(writer_test_paths)
    if len(test_labels) == 0:
        errors += 1
        print("Error", errors)
        continue
    # base = "../../forms"
    # features = extract_features("{}/{}.png".format(base, train_images[0]))
    # for image in train_images[1:]:
    #     features = np.append(features, extract_features("{}/{}.png".format(base, image)), axis=0)
    # for image in test_images:
    #     features = np.append(features, extract_features("{}/{}.png".format(base, image)), axis=0)
    # test_images_count = len(test_images)
    #
    # features = feature_extractor.apply_pca(features)
    # classifier = Classifier()
    # classifier.train(features[:-test_images_count], train_labels)
    # print(train_labels)
    # print(classifier.classify(features[-test_images_count:]))
    # print(np.array(test_labels))
    print()



# for i in range(10, 11):
#     path = "../data/" + str(i)
#     features = extract_features(path + "/1/1.png")
#     features = np.append(features, extract_features(path + "/1/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/2/1.png"), axis=0)
#     features = np.append(features, extract_features(path + "/2/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/3/1.png"), axis=0)
#     features = np.append(features, extract_features(path + "/3/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/test.png"), axis=0)
#
#     # cv2.imshow('Original Image', img)
#     print(type(features))
#     print(features.shape)
#     print(features)
#     features = feature_extractor.apply_pca(features)
#     print(type(features))
#     print(features.shape)
#     print(features)
#
#     classifier = Classifier(3)
#     classifier.train(features[:-1], [1, 1, 2, 2, 3, 3])
#     print(classifier.classify([features[-1]]))
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
