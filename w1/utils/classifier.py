import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(n_neighbors, distance):
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        n_jobs=-1,
        metric=distance
    )
    return knn


def cluster(x, n_clusters):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        verbose=False,
        batch_size=n_clusters * 20,
        compute_labels=False,
        reassignment_ratio=10 ** -4,
        random_state=42
    )
    kmeans.fit(x)
    return kmeans


class Classifier:
    def __init__(self, k, n, d):
        self.k = k
        self._clf = knn_classifier(n, d)
        self._codebook = None

    def _aggregate(self, descriptors):
        # For each train image, project each keypoint descriptor to its closest visual word.
        # Each image is represented by the frequency of each visual word.
        visual_words = np.empty((len(descriptors), self.k), dtype=np.float32)
        for i, des in enumerate(descriptors):
            words = self._codebook.predict(des)
            histogram = np.bincount(words, minlength=self.k)
            histogram = normalize(histogram.reshape(1, -1), norm='l2')  # normalize histogram
            visual_words[i, :] = histogram
        return visual_words

    def train(self, train_descriptors, train_labels):
        # Compute a k-means clustering on the descriptor space.
        self._codebook = cluster(np.vstack(train_descriptors), self.k)
        # Compute the train visual words.
        train_visual_words = self._aggregate(train_descriptors)
        # Build a k-nn classifier and train it with the train descriptors.
        self._clf.fit(train_visual_words, train_labels)

    def predict(self, test_descriptors):
        # Compute the test visual words.
        test_visual_words = self._aggregate(test_descriptors)
        # Predict the class labels for the test data.
        predicted = self._clf.predict(test_visual_words)
        return predicted

    def test(self, test_descriptors, test_labels):
        # Compute the test visual words.
        test_visual_words = self._aggregate(test_descriptors)
        # Compute accuracy of the model.
        accuracy = self._clf.score(test_visual_words, test_labels)
        return accuracy
