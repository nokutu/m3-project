import argparse
import multiprocessing.dummy as mp
import os

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

from timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='data/MIT_split/test')
    parser.add_argument('--n_features', type=int, default=300)
    parser.add_argument('--n_clusters', type=int, default=128)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--distance', type=str, default='euclidean')
    return parser.parse_args()


def load_dataset(path):
    filenames, labels = [], []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if not os.path.isdir(label_path):
            continue
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            if not image_path.endswith('.jpg'):
                continue
            filenames.append(image_path)
            labels.append(label)
    return filenames, labels


def extract_descriptors(sift, filenames):
    def _worker(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        kps, des = sift.detectAndCompute(img, None)
        return des

    with mp.Pool(processes=4) as p:
        descriptors = p.map(_worker, filenames)

    return descriptors


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


def classifier(n_neighbors, distance):
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        n_jobs=-1,
        metric=distance
    )
    return knn


def main():
    args = parse_args()

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    # Create a SIFT object detector and descriptor.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=args.n_features)

    # Compute the SIFT descriptors for all the train images.
    with Timer('extract train descriptors'):
        train_descriptors = extract_descriptors(sift, train_filenames)

    # Compute a k-means clustering on the descriptor space.
    k = args.n_clusters
    codebook = cluster(np.vstack(train_descriptors), k)

    # For each train image, project each keypoint descriptor to its closest visual word.
    # Each image is represented by the frequency of each visual word.
    train_visual_words = np.empty((len(train_descriptors), k), dtype=np.float32)
    for i, des in enumerate(train_descriptors):
        words = codebook.predict(des)
        train_visual_words[i, :] = np.bincount(words, minlength=k)

    # Build a k-nn classifier and train it with the train descriptors.
    clf = classifier(args.n_neighbors, args.distance)
    clf.fit(train_visual_words, train_labels)

    # Compute the test descriptors.
    with Timer('extract test descriptors'):
        test_descriptors = extract_descriptors(sift, test_filenames)

    # Compute the test visual words.
    test_visual_words = np.empty((len(test_descriptors), k), dtype=np.float32)
    for i, des in enumerate(test_descriptors):
        words = codebook.predict(des)
        test_visual_words[i, :] = np.bincount(words, minlength=k)

    # Compute accuracy of the model.
    accuracy = clf.score(test_visual_words, test_labels)
    print('accuracy: {:.6f}'.format(accuracy))


if __name__ == '__main__':
    with Timer('total time'):
        main()
