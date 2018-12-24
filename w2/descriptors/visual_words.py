from typing import List
import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from functional import seq

from model.picture import Picture


class BoWTransformer:

    def __init__(self, n_clusters: int = 512):
        self.k = n_clusters
        self._codebook = MiniBatchKMeans(
            n_clusters=n_clusters,
            verbose=False,
            batch_size=n_clusters * 20,
            compute_labels=False,
            reassignment_ratio=10 ** -4,
            random_state=42)

    def fit(self, pictures: List[Picture], y=None):
        descriptors = [p.descriptors for p in pictures]
        #descriptors = random.sample(descriptors, int(len(descriptors)/2))
        self._codebook.fit(np.vstack(descriptors))
        return self

    def transform(self, pictures: List[Picture]):
        descriptors = [p.descriptors for p in pictures]
        visual_words = np.empty((len(descriptors), self.k), dtype=np.float32)
        for i, des in enumerate(descriptors):
            words = self._codebook.predict(des)
            histogram = np.bincount(words, minlength=self.k)
            histogram = self._normalize(histogram)  # normalize histogram
            visual_words[i, :] = histogram
        return visual_words

    @staticmethod
    def _normalize(x: np.ndarray):
        return normalize(x.reshape(1, -1), norm='l2')


class SpatialPyramid(BoWTransformer):

    def __init__(self, n_clusters: int = 512, levels: int = 1):
        super().__init__(n_clusters)
        self.levels = levels

    def transform(self, pictures: List[Picture]):
        blocks = seq.range(1, self.levels + 1).map(lambda l: l ** 2).sum()
        visual_words = np.empty((len(pictures), blocks, self.k), dtype=np.float32)
        for i, picture in enumerate(pictures):
            words = self._codebook.predict(picture.descriptors)
            pos = 0
            for level in range(1, self.levels + 1):
                word_sets = self._descriptor_sets(level, picture, words)
                for word_set in word_sets:
                    histogram = np.bincount(word_set, minlength=self.k)
                    visual_words[i, pos, :] = self._normalize(histogram)
                    pos += 1
        return visual_words.reshape((len(pictures), -1))

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
