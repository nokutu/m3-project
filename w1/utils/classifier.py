from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from timer import Timer
from utils.extract_descriptors import extract_descriptors


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


def classifier(train_descriptors, train_labels, test_descriptors, test_labels, k, n, d):
    # Compute a k-means clustering on the descriptor space.
    codebook = cluster(np.vstack(train_descriptors), k)

    # For each train image, project each keypoint descriptor to its closest visual word.
    # Each image is represented by the frequency of each visual word.
    train_visual_words = np.empty((len(train_descriptors), k), dtype=np.float32)
    for i, des in enumerate(train_descriptors):
        words = codebook.predict(des)
        train_visual_words[i, :] = np.bincount(words, minlength=k)

    # Build a k-nn classifier and train it with the train descriptors.
    clf = knn_classifier(n, d)
    clf.fit(train_visual_words, train_labels)


    # Compute the test visual words.
    test_visual_words = np.empty((len(test_descriptors), k), dtype=np.float32)
    for i, des in enumerate(test_descriptors):
        words = codebook.predict(des)
        test_visual_words[i, :] = np.bincount(words, minlength=k)

    # Compute accuracy of the model.
    accuracy = clf.score(test_visual_words, test_labels)
    return accuracy
