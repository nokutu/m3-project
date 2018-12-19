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

    def fit(self):
        self.cluster = SpatialPyramid.get_cluster(self.n_clusters)

        pass

    def predict(self):
        pass

    def _compute(self, pictures: List[Picture]):
        i = 0
        s = 0
        for i in range(1, self.levels + 1):
            s += i ** 2
        train_visual_words = np.empty((len(pictures), self.n_clusters*s), dtype=np.float32)
        for level in range(1, self.levels+1):
            a = self._descriptor_sets(level, pictures)

            for des_set in a:
                for des in des_set:
                    words = self.cluster.predict(des)
                    train_visual_words[i, :] = np.bincount(words, minlength=self.n_clusters)
                    i += 1

        return train_visual_words

    def _descriptor_sets(self, level, pictures):
        if level == 1:
            return pictures.descriptors
        else:
            for k in keypoints:
        pass

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
