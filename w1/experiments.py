from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt

from bovw import run


def run_experiment(method=['sift'], n_features=[300], step_size=[16], n_clusters=[128], n_neighbors=[5], distance=['euclidean']):
    args = Namespace(train_path='../data/MIT_split/train',
                     test_path='../data/MIT_split/test',
                     method=method,
                     n_features=n_features,
                     step_size=step_size,
                     n_clusters=n_clusters,
                     n_neighbors=n_neighbors,
                     distance=distance)
    return run(args)


def experiment_1():
    """
    Test different amounts of local features.
    """
    results = run_experiment(n_features=[100, 1000, 10])
    plt.scatter(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


def experiment_2():
    """
    SIFT vs DenseSIFT.
    """
    results = run_experiment(method=['sift', 'dense_sift'])
    x = np.arange(2)
    plt.bar(x, results['accuracy'])
    plt.xticks(x, results['method'])
    plt.ylabel('accuracy')
    plt.show()


def experiment_3():
    """
    Test different amounts of codebook sizes k.
    """
    results = run_experiment(n_clusters=[80, 170, 10])
    plt.scatter(results['n_clusters'], results['accuracy'])
    plt.xlabel('n_clusters')
    plt.ylabel('accuracy')
    plt.show()


def experiment_4():
    """
    Test different values of k for the k-nn classifier.
    """
    results = run_experiment(n_neighbors=[3, 21, 10])
    plt.scatter(results['n_neighbors'], results['accuracy'])
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.show()


def experiment_5():
    """
    Test different distances for the k-nn classifier.
    """
    results = run_experiment(distance=['euclidean', 'manhattan', 'chebyshev'])
    x = np.arange(3)
    plt.bar(x, results['accuracy'])
    plt.xticks(x, results['distance'])
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
