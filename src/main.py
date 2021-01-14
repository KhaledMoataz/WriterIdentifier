#!/usr/bin/env python3

from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
from preprocessor import preProcessor


def generate_random_testcase(writers_list):
    writers, _ = utils.get_random_indices(writers_list, 3)
    train_images = []
    test_image = None
    test_truth = None
    got_test = False
    for idx in range(len(writers)):
        images, valid = utils.get_random_indices(writers_list[writers[idx]], 2 + (1 - got_test))
        if valid and not got_test:
            got_test = True
            test_image = writers_list[writers[idx]][images[2]]
            test_truth = idx
            images.pop()
        for img_idx in images:
            train_images.append(writers_list[writers[idx]][img_idx])
    if test_image is None:
        return -1
    for idx in range(len(train_images)):
        path = "../data/forms/{}.png".format(train_images[idx])
        train_images[idx] = preProcessor(path)

    feature_extractor = FeatureExtractor.FeatureExtractor(2)
    features_list = feature_extractor.extract_features(test_image)
    for image in range(train_images):
        features_list.append(feature_extractor.extract_features(image), axis=0)

    pca_features = feature_extractor.apply_pca(features_list)
    classifier = Classifier()
    label_list = [0, 0, 1, 1, 2, 2]
    classifier.train(pca_features[1:], label_list)
    calssification_result = classifier.classify([pca_features[0]])
    return calssification_result == test_truth


if __name__ == '__main__':
    num_cases = 1
    num_correct_predictions = 0
    writers_list = utils.read_ascii()
    for _ in range(num_cases):
        ret = -1
        while ret == -1:
            ret = generate_random_testcase(writers_list)
        num_correct_predictions += ret
    print("Number of Cases: {}".format(num_cases))
    print("Model Accuracy: {}%".format(100 * num_correct_predictions / num_cases))
