from typing import List

import numpy as np
from functional import seq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans

from model.picture import Picture


class BoWTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters: int = 500, n_samples: int = 10000, norm: str = 'l2'):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.norm = norm

        self._codebook = None
        np.random.seed(42)

    def fit(self, pictures: np.ndarray, y=None):
        self._codebook = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters * 3,
            compute_labels=False,
            reassignment_ratio=10 ** -4,
            random_state=42)

        descriptors = np.vstack([p[1] for p in pictures])
        descriptors = descriptors[np.random.choice(descriptors.shape[0], self.n_samples, replace=False), :]
        self._codebook.fit(descriptors)
        return self

    def transform(self, pictures: np.ndarray):
        descriptors = [p[1] for p in pictures]
        visual_words = np.empty((len(descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(descriptors):
            words = self._codebook.predict(des)
            histogram = np.bincount(words, minlength=self.n_clusters)
            histogram = self._normalize(histogram)  # normalize histogram
            visual_words[i, :] = histogram
        return visual_words

    def _normalize(self, x: np.ndarray, alpha: float = 0.5):
        if self.norm == 'l1':
            norm = np.linalg.norm(x, ord=1)
            x_norm = x / norm
        elif self.norm == 'l2':
            norm = np.linalg.norm(x, ord=2)
            x_norm = x / norm
        elif self.norm == 'power':
            # https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf
            x = np.sign(x) * np.abs(x) ** alpha
            norm = np.linalg.norm(x, ord=2)
            x_norm = x / norm
        else:
            raise ValueError("'{}' is not a supported norm".format(self.norm))
        return x_norm


class SpatialPyramid(BoWTransformer):

    def __init__(self, n_clusters: int = 500, n_samples: int = 10000, norm: str = 'l2', levels: int = 2):
        super().__init__(n_clusters, n_samples, norm)
        self.levels = levels

        #print('{}: {}'.format(self.__class__.__name__, vars(self)))

    def transform(self, pictures: np.ndarray):
        n_blocks = seq.range(self.levels).map(lambda l: 4 ** l).sum()
        visual_words = np.empty((len(pictures), n_blocks * self.n_clusters), dtype=np.float32)
        for i, picture in enumerate(pictures):
            words = self._codebook.predict(picture[1])
            j = 0
            for l in range(self.levels):
                word_sets = self._descriptor_sets(l, picture)
                w = 1 / 2 ** (self.levels - l)  # descriptors at finer resolutions are weighted more
                for inds in word_sets:
                    histogram = np.bincount(words[inds], minlength=self.n_clusters)
                    histogram = self._normalize(histogram) * w
                    visual_words[i, j:j + self.n_clusters] = histogram
                    j += self.n_clusters
        return visual_words

    @staticmethod
    def _descriptor_sets(level: int, picture: np.ndarray):
        h, w = picture[2]
        block_h = h / 2 ** level
        block_w = w / 2 ** level
        word_sets = [[] for _ in range(4 ** level)]
        for idx, kp in enumerate(picture[0]):
            i = int(np.floor(kp[1] / block_h))
            j = int(np.floor(kp[0] / block_w))
            word_sets[i * (2 ** level) + j].append(idx)
        return word_sets
