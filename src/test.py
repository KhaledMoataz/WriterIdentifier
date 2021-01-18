import numpy as np
import preprocessor
from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils

feature_extractor = FeatureExtractor.FeatureExtractor(6)
classifier = Classifier()

LABELS = [1, 1, 2, 2, 3, 3]

train_images_names = ['a02-111', 'a02-106', 'a05-013', 'a05-058', 'g06-026l', 'g06-050l']
test_image_name = 'a01-000u'
train_images = [preprocessor.preProcessor(utils.read_image("../../forms/" + image_name + ".png")) for image_name in
                train_images_names]
test_image = preprocessor.preProcessor(utils.read_image("../../forms/" + test_image_name + ".png"))
features = feature_extractor.extract_features(test_image)
for image in train_images:
    features = np.append(features, feature_extractor.extract_features(image), axis=0)
features = feature_extractor.apply_pca(features)
classifier.clear()
classifier.train(features[1:], LABELS)
print(classifier.classify(features[0:1])[0])
utils.show_results([2], [1], zip(train_images_names, train_images), [(test_image_name, test_image)], LABELS)
