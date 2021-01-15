from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import preprocessing
import numpy as np

class Classifier:
    def __init__(self, k=1):
        self.k = k
        self.model = NearestCentroid()
        # self.model = KNeighborsClassifier(n_neighbors=k)
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
        # self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model = NearestCentroid()
