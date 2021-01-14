from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


class Classifier:
    def __init__(self, k=1):
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.normalizer = preprocessing.Normalizer(norm='l2')

    def __normalize(self, features):
        return self.normalizer.transform(features)

    def train(self, features, labels):
        self.model.fit(self.__normalize(features), labels)

    def classify(self, features):
        return self.model.predict(self.__normalize(features))
