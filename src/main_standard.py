import numpy as np
import preprocessor
from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
import timeit
prediction_file = open("results.txt", "w")
time_file = open("time.txt", "w")

LABELS = [1, 1, 2, 2, 3, 3]

feature_extractor = FeatureExtractor.FeatureExtractor(4)
classifier = Classifier()

for test_case_number in range(1, 11):
    test_case_number = "{:02d}".format(test_case_number)
    base = "../data/{}".format(test_case_number)
    test_image = utils.read_image("{}/test.png".format(base))
    train_images = []
    for writer in range(1, 4):
        for page in range(1, 3):
            train_images.append(utils.read_image("{}/{}/{}.png".format(base, writer, page)))
    start_time = timeit.default_timer()
    # Preprocessing images
    test_image = preprocessor.preProcessor(test_image)
    train_images = [preprocessor.preProcessor(image) for image in train_images]
    # Extracting features
    features = feature_extractor.extract_features(test_image)
    for image in train_images:
        features = np.append(features, feature_extractor.extract_features(image), axis=0)
    classifier.clear()
    classifier.train(features[1:], LABELS)
    predicted = classifier.classify(features[0:1])[0]
    execution_time = round(timeit.default_timer() - start_time, 2)
    prediction_file.write(str(predicted) + '\n')
    time_file.write(str(execution_time) + '\n')
prediction_file.close()
time_file.close()


