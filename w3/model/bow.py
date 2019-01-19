import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans


class BoWTransformer(BaseEstimator, TransformerMixin):

    _codebook: MiniBatchKMeans

    def __init__(self, n_clusters: int = 500, n_samples: int = 10000, norm: str = 'l2'):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.norm = norm

        self._codebook = None

    def fit(self, descriptors: np.ndarray, y=None):
        np.random.seed(42)

        self._codebook = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters * 3,
            compute_labels=False,
            reassignment_ratio=10 ** -4)

        s = descriptors.shape
        descriptors = descriptors.reshape(s[0]*s[1], s[2])
        descriptors = descriptors[np.random.choice(descriptors.shape[0], self.n_samples, replace=False), :]
        self._codebook.fit(descriptors)
        return self

    def transform(self, descriptors: np.ndarray):
        visual_words = np.empty((descriptors.shape[0], self.n_clusters), dtype=np.float32)
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
