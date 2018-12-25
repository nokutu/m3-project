import random
from typing import List

import numpy as np
from functional import seq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from model.picture import Picture


class BoWTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters: int = 512, norm: str = 'l2'):
        self.n_clusters = n_clusters
        self.norm = norm

        self._codebook = None

    def fit(self, pictures: List[Picture], y=None):
        self._codebook = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters * 3,
            compute_labels=False,
            reassignment_ratio=10 ** -4,
            random_state=42)

        descriptors = [p.descriptors for p in pictures]
        descriptors = random.sample(descriptors, min(len(descriptors), 10000))
        self._codebook.fit(np.vstack(descriptors))
        return self

    def transform(self, pictures: List[Picture]):
        descriptors = [p.descriptors for p in pictures]
        visual_words = np.empty((len(descriptors), self.n_clusters), dtype=np.float32)
        for i, des in enumerate(descriptors):
            words = self._codebook.predict(des)
            histogram = np.bincount(words, minlength=self.n_clusters)
            histogram = self._normalize(histogram)  # normalize histogram
            visual_words[i, :] = histogram
        return visual_words

    def _normalize(self, x: np.ndarray):
        return normalize(x.reshape(1, -1), norm=self.norm).ravel()


class SpatialPyramid(BoWTransformer):

    def __init__(self, n_clusters: int = 512, levels: int = 2):
        super().__init__(n_clusters)
        self.levels = levels

    def transform(self, pictures: List[Picture]):
        blocks = seq.range(1, self.levels + 1).map(lambda l: l ** 2).sum()
        visual_words = np.empty((len(pictures), blocks * self.n_clusters), dtype=np.float32)
        for i, picture in enumerate(pictures):
            words = self._codebook.predict(picture.descriptors)
            pos = 0
            for level in range(1, self.levels + 1):
                word_sets = self._descriptor_sets(level, picture, words)
                for word_set in word_sets:
                    histogram = np.bincount(word_set, minlength=self.n_clusters)
                    visual_words[i, pos:pos + self.n_clusters] = self._normalize(histogram)
                    pos += self.n_clusters
        return visual_words

    @staticmethod
    def _descriptor_sets(level: int, picture: Picture, words: np.ndarray):
        if level == 1:
            return [words]
        else:
            block_h = picture.size[0] / level
            block_w = picture.size[1] / level
            word_sets = [[] for _ in range(level ** 2)]
            for kp, word in zip(picture.keypoints, words):
                i = int(kp[1] / block_h)
                j = int(kp[0] / block_w)
                word_sets[i * level + j].append(word)
            return word_sets
