#!/usr/bin/env python3

from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
from preprocessor import preProcessor
import numpy as np
import timeit


def generate_random_testcase(writers_list):
    writers = utils.get_random_indices(writers_list, 3)
    if len(writers) < 3:
        return -1
    train_images = []
    test_image = None
    test_truth = None
    got_test = False
    for idx in range(len(writers)):
        images = utils.get_random_indices(writers_list[writers[idx]], 2 + (1 - got_test))
        num = len(images)
        if num == 3:
            got_test = True
            test_image = writers_list[writers[idx]][images[2]]
            test_truth = idx
            images.pop()
        elif num < 2:
            return -1
        for img_idx in images:
            train_images.append(writers_list[writers[idx]][img_idx])
    if not got_test:
        return -1

    processed_images = [preProcessor("../data/forms/{}.png".format(test_image))]
    for image_name in train_images:
        path = "../data/forms/{}.png".format(image_name)
        processed_images.append(preProcessor(path))

    feature_extractor = FeatureExtractor.FeatureExtractor(2)
    features_list = feature_extractor.extract_features(processed_images[0])
    for idx in range(1, len(processed_images)):
        features_list = np.append(features_list, feature_extractor.extract_features(processed_images[idx]), axis=0)

    pca_features = feature_extractor.apply_pca(features_list)
    classifier = Classifier()
    labels_list = [0, 0, 1, 1, 2, 2]

    classifier.train(pca_features[1:], labels_list)
    classification_result = classifier.classify([pca_features[0]])
    if classification_result != test_truth:
        print("Train Images: ", train_images)
        print("Test Image: ", test_image)
        print("Test Truth: {}, Classification Result: {}".format(test_truth, classification_result))
        return False
    return True


if __name__ == '__main__':
    num_cases = 100
    num_correct_predictions = 0
    writers_list = utils.read_ascii()
    for case in range(num_cases):
        ret = -1
        start = None
        end = None
        while ret == -1:
            start = timeit.default_timer()
            ret = generate_random_testcase(writers_list)
            end = timeit.default_timer()
        num_correct_predictions += ret
        print("Case: {}, Result: {}, Time:{}".format(case + 1, ret, end - start))
    print("Number of Cases: {}".format(num_cases))
    print("Model Accuracy: {}%".format(100 * num_correct_predictions / num_cases))
