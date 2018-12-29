from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt

from descriptors.histogram_intersection_kernel import histogram_intersection_kernel
from main import main


def run_experiment(param_grid: dict):
    args = Namespace(train_path='../data/MIT_split/train',
                     test_path='../data/MIT_split/test',
                     cache_path='../data/cache')
    return main(args, param_grid)


def experiment_1():
    """
    Test different kernels and penalty parameter of the error term "C" for the classifier.
    """
    param_grid = {
        'classifier__kernel': ['linear', 'rbf', 'sigmoid', histogram_intersection_kernel],
        'classifier__C': np.logspace(-4, 4, 9),
    }

    results = run_experiment(param_grid)

    # plt.plot(results['n_features'], results['accuracy'])
    # plt.xlabel('n_features')
    # plt.ylabel('accuracy')
    # plt.show()


def experiment_2():
    """
    Test different kernels and kernel coefficients for the classifier.
    """
    param_grid = {
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__gamma': np.logspace(-4, 4, 9)
    }

    results = run_experiment(param_grid)
    plt.plot(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


def experiment_3():
    """
    Test the classifier using different amounts of samples randomly selected form the descriptors,
    which were obtained using dense_sift with different scale factors.
    """
    param_grid = {
        'transformer__samples': np.linspace(10000, 100000, 5),
    }

    results = run_experiment(param_grid)
    plt.plot(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


def experiment_4():
    """
    Test different normalization for the descriptors.
    """
    param_grid = {
        'transformer__norm': ["l1", "l2", "power"],
    }

    results = run_experiment(param_grid)
    plt.plot(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


def experiment_5():
    """
    Test different number of levels for the spatial pyramid that takes into account the location of the
    descriptors to generate a general image descriptor.
    """
    param_grid = {
        'transformer__levels': np.linspace(1, 3, 3),
    }

    results = run_experiment(param_grid)
    plt.plot(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


def experiment_6():
    """
    Test different number of clusters, multiples of 2, for the min_batch k-means.
    """
    param_grid = {
        'transformer__n_cluster': np.logspace(8, 11, 4, base=2),
    }

    results = run_experiment(param_grid)
    plt.plot(results['n_features'], results['accuracy'])
    plt.xlabel('n_features')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    experiment_1()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # experiment_6()
    # experiment_7()
    # experiment_8()
