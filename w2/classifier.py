from typing import List

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from multiprocessing import Pool

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 kernel="rbf",
                 n_clusters=128,
                 C=1,
                 n_neighbors=16,
                 normalization="L2",
                 distance="euclidean",
                 classifier="knn"):

        self.kernel = kernel
        self.n_clusters = n_clusters
        self.C = C
        self.n_neighbors = n_neighbors
        self.normalization = normalization
        self.distance = distance
        self.classifier = classifier

        self.clf = None
        self.scaler = None
        self.cluster = None

    def fit(self, train_descriptors: np.ndarray, labels: List[str]):
        self.cluster = self.get_cluster(self.n_clusters)
        self.scaler = StandardScaler()

        if self.classifier == 'svm':
            self.clf = SVC(C=self.C, kernel=self.kernel)
        elif self.classifier == 'knn':
            self.clf = self.get_knn_classifier(self.n_neighbors, self.distance)

        self.cluster.fit(np.vstack(train_descriptors))

        train_visual_words = np.empty((len(train_descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(train_descriptors):
            words = self.cluster.predict(des)
            train_visual_words[i, :] = np.bincount(words, minlength=self.n_clusters)

        self.clf.fit(train_visual_words, labels)

    def predict(self, test_descriptors: np.ndarray, y=None):
        test_visual_words = np.empty((len(test_descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(test_descriptors):
            words = self.cluster.predict(des)
            test_visual_words[i, :] = np.bincount(words, minlength=self.n_clusters)

        return test_visual_words

    def score(self, test_descriptors: np.ndarray, labels=None, sample_weight=None):
        return self.clf.score(self.predict(test_descriptors), labels)

    @staticmethod
    def get_cluster(n_clusters: int):
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            verbose=False,
            batch_size=n_clusters * 20,
            compute_labels=False,
            reassignment_ratio=10 ** -4,
            random_state=42
        )

    @staticmethod
    def get_knn_classifier(n_neighbors, distance):
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1,
            metric=distance
        )
        return knn





