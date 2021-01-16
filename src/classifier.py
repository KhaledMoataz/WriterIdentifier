import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestCentroid


class Classifier:
    def __init__(self):
        self.model = NearestCentroid()
        self.normalizer = preprocessing.Normalizer(norm='l2')

    def __normalize(self, features):
        eps = 1e-7
        features /= features.sum() + eps
        features = np.sqrt(features)
        return self.normalizer.transform(features)

    def train(self, features, labels):
        self.model.fit(self.__normalize(features), labels)

    def classify(self, features):
        return self.model.predict(self.__normalize(features))

    def clear(self):
        self.model = NearestCentroid()
