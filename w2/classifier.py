import random
from typing import List

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import cv2
from multiprocessing import Pool

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from w2.utils.timer import Timer


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 step_size=16,
                 kernel="rbf",
                 n_clusters=128,
                 C=1,
                 n_neighbors=16,
                 normalization="L2",
                 distance="euclidean",
                 classifier="knn"):

        self.step_size = step_size
        self.kernel = kernel
        self.n_clusters = n_clusters
        self.C = C
        self.n_neighbors = n_neighbors
        self.normalization = normalization
        self.distance = distance
        self.classifier = classifier

        self.sift = None
        self.clf = None
        self.scaler = None
        self.cluster = None

    def fit(self, filenames: List[str], labels: List[str]):
        self.sift = DenseSIFT(self.step_size)
        self.cluster = self.get_cluster(self.n_clusters)
        self.scaler = StandardScaler()

        if self.classifier == 'svm':
            self.clf = SVC(C=self.C, kernel=self.kernel)
        elif self.classifier == 'knn':
            self.clf = self.get_knn_classifier(self.n_neighbors, self.distance)

        with Timer('Extracting fit descriptors'):
            train_descriptors = self.sift.compute(filenames)

        self.cluster.fit(np.vstack(train_descriptors))

        train_visual_words = np.empty((len(train_descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(train_descriptors):
            words = self.cluster.predict(des)
            train_visual_words[i, :] = np.bincount(words, minlength=self.n_clusters)

        self.clf.fit(train_visual_words, labels)

    def predict(self, filenames: List[str], y=None):
        test_descriptors = self.sift.compute(filenames)
        test_visual_words = np.empty((len(test_descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(test_descriptors):
            words = self.cluster.predict(des)
            test_visual_words[i, :] = np.bincount(words, minlength=self.n_clusters)

        return test_visual_words

    def score(self, filenames: List[str], labels=None, sample_weight=None):
        return self.clf.score(self.predict(filenames), labels)

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


_images = dict()  # cache images


class DenseSIFT:

    def __init__(self, step_size: int):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._step_size = step_size

    def compute(self, filenames: List[str]):
        def _worker(filename):
            if filename not in _images:
                _images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            des = self._compute(_images[filename])

            return des

        return [_worker(filename) for filename in filenames]

    def _compute(self, img):
        kps = []
        for x in range(0, img.shape[1], self._step_size):
            for y in range(0, img.shape[0], self._step_size):
                size = self._step_size * random.uniform(1, 3)
                kp = cv2.KeyPoint(x, y, size)
                kps.append(kp)
        _, des = self._sift.compute(img, kps)
        return des
