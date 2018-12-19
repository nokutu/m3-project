import cv2
import random
from typing import List
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans

from w2.model.picture import Picture


class SpatialPyramid(BaseEstimator, ClusterMixin):

    def __init__(self,
                 n_clusters=128,
                 levels=1):
        self.n_clusters = n_clusters
        self.levels = levels

        self.cluster = None

    def fit(self, pictures, y=None):
        return self._compute(pictures)

    def predict(self, pictures, y=None):
        return self._compute(pictures)

    def _compute(self, pictures: List[Picture]):
        s = 0
        for i in range(1, self.levels + 1):
            s += i ** 2

        train_visual_words = np.empty((len(pictures), self.n_clusters * s), dtype=np.float32)

        for i, picture in enumerate(pictures):
            words = self.cluster.predict(picture.descriptors)
            pos = 0

            for level in range(1, self.levels + 1):
                word_sets = self._descriptor_sets(level, pictures, words)

                for word_set in word_sets:
                    train_visual_words[i, pos:pos + self.n_clusters] = np.bincount(word_set, minlength=self.n_clusters)
                    pos += self.n_clusters

        return train_visual_words

    @staticmethod
    def _descriptor_sets(level, pictures, words):
        if level == 1:
            return words
        else:
            res = [[] for _ in range(level ** 2)]

            for picture in pictures:
                width_step = picture.size[0] / level
                height_step = picture.size[1] / level

                for word, kp in zip(words, picture.keypoints):
                    w = kp[0] // width_step
                    h = kp[1] // height_step
                    res[w * level + h].append(word)

            return res

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
