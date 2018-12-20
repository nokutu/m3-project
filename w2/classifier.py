from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Classifier(BaseEstimator, ClassifierMixin):
    clf: Union[SVC, KNeighborsClassifier]

    def __init__(self,
                 kernel="rbf",
                 C=1,
                 n_neighbors=16,
                 normalization="L2",
                 distance="euclidean",
                 classifier="knn",
                 ):

        self.kernel = kernel
        self.C = C
        self.distance = distance
        self.classifier = classifier
        self.n_neighbors = n_neighbors

        self.clf = None

    def fit(self, x: np.ndarray, labels: List[str]):
        if self.classifier == 'svm':
            self.clf = SVC(C=self.C, kernel=self.kernel)
        elif self.classifier == 'knn':
            self.clf = self.get_knn_classifier(self.n_neighbors, self.distance)
        else:
            raise TypeError('Invalid classifier, must be svm or knn')

        self.clf.fit(x, labels)

    def predict(self, x: np.ndarray, y=None):
        return self.clf.predict(x)

    def score(self, x: np.ndarray, labels=None, sample_weight=None):
        return self.clf.score(x, labels)

    @staticmethod
    def get_knn_classifier(n_neighbors, distance):
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1,
            metric=distance
        )
        return knn
