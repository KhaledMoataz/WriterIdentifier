import numpy as np
import preprocessor
from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
import timeit

LABELS = [1, 1, 2, 2, 3, 3]

feature_extractor = FeatureExtractor.FeatureExtractor(4)
classifier = Classifier()
#
# for test_case_number in range(1, 2):
#     test_case_number = "{:02d}".format(test_case_number)
#     base = "../data/{}".format(test_case_number)
#     test_image = utils.read_image("{}/test.png".format(base))
#     train_images = []
#     for writer in range(1, 4):
#         for page in range(1, 3):
#             train_images.append(utils.read_image("{}/{}/{}.png".format(base, writer, page)))
#     start_time = timeit.default_timer()
#     # Preprocessing images
#     test_image = preprocessor.preProcessor(test_image)
#     train_images = [preprocessor.preProcessor(image) for image in train_images]
#     # Extracting features
#     features = feature_extractor.extract_features(test_image)
#     for image in train_images:
#         features = np.append(features, feature_extractor.extract_features(image), axis=0)
#     classifier.clear()
#     classifier.train(features[1:], LABELS)
#     print(classifier.classify(features[0:1])[0])
#     print(timeit.default_timer() - start_time)
#
# train_images_names = ['b01-053', 'b01-049', 'g07-044', 'g07-047', 'h01-007', 'h01-000']
# test_image_name = 'g07-038'
train_images_names = ['p06-042', 'p06-047', 'f02-033', 'f02-017', 'g06-018r', 'g06-042r']
test_image_name = 'p06-096'
train_images = [preprocessor.preProcessor(utils.read_image("../../forms/" + image_name + ".png")) for image_name in train_images_names]
test_image = preprocessor.preProcessor(utils.read_image("../../forms/" + test_image_name + ".png"))
features = feature_extractor.extract_features(test_image)
for image in train_images:
    features = np.append(features, feature_extractor.extract_features(image), axis=0)
classifier.clear()
classifier.train(features[1:], LABELS)
print(classifier.classify(features[0:1])[0])
