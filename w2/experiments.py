from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt

from descriptors.histogram_intersection_kernel import histogram_intersection_kernel
from main import main


def run_experiment(param_grid: dict, plot_confusion=False):
    args = Namespace(train_path='../data/MIT_split/train',
                     test_path='../data/MIT_split/test',
                     cache_path='../.cache')
    return main(args, param_grid, plot_confusion=plot_confusion)


def experiment_1():
    """
    Test different amounts of samples randomly selected form the descriptors,
    which were obtained using dense_sift with different scale factors.
    """
    param_grid = {
        'transformer__n_samples': np.linspace(10000, 100000, 5, dtype=int),
    }
    results = run_experiment(param_grid)

    results.plot.line(x='param_transformer__n_samples', y='mean_test_score')
    plt.xlabel('samples')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()


def experiment_2():
    """
    Test different codebook sizes, multiples of 2, for the min_batch k-means.
    """
    param_grid = {
        'transformer__n_clusters': np.logspace(7, 10, 8, base=2, dtype=int),
        'transformer__n_levels': [1]
    }
    results = run_experiment(param_grid)

    results.plot.line(x='param_transformer__n_clusters', y='mean_test_score')
    plt.xlabel('n_clusters')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()


def experiment_3():
    """
    Test different normalization for the descriptors.
    """
    param_grid = {
        'transformer__norm': ['l1', 'l2', 'power'],
    }
    results = run_experiment(param_grid)

    # Colormap needed until a bug is fixed in next version of pandas.
    results.plot.bar(x='param_transformer__norm', y='mean_test_score', colormap='jet')
    plt.xlabel('norm')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()


def experiment_4():
    """
    Test different number of levels for the spatial pyramid that takes into account the location of the
    descriptors to generate a general image descriptor.
    """
    param_grid = {
        'transformer__n_levels': np.linspace(1, 3, 3, dtype=int),
    }
    results = run_experiment(param_grid)

    # Colormap needed until a bug is fixed in next version of pandas.
    results.plot.bar(x='param_transformer__n_levels', y='mean_test_score', colormap='jet')
    plt.xlabel('n_levels')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()


def experiment_5():
    """
    Test different kernels and penalty parameter 'C' for the classifier.

    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    """
    param_grid = {
        'classifier__kernel': ['linear', 'rbf', 'sigmoid', histogram_intersection_kernel],
        'classifier__C': np.logspace(-3, 15, 5, base=2),
    }

    results = run_experiment(param_grid)
    results.loc[results.param_classifier__kernel == histogram_intersection_kernel, 'param_classifier__kernel'] = \
        "histogram_intersection"

    results.pivot(index='param_classifier__C', columns='param_classifier__kernel', values='mean_test_score') \
        .plot.line(logx=True)

    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    plt.show()


def experiment_6():
    """
    Test different kernels and kernel coefficients 'gamma' for the classifier.

    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    """
    param_grid = {
        'classifier__kernel': ['linear', 'rbf', 'sigmoid', histogram_intersection_kernel],
        'classifier__gamma': np.logspace(-15, 3, 5, base=2)
    }

    results = run_experiment(param_grid)
    results.loc[results.param_classifier__kernel == histogram_intersection_kernel, 'param_classifier__kernel'] = \
        "histogram_intersection"

    results.pivot(index='param_classifier__gamma', columns='param_classifier__kernel', values='mean_test_score') \
        .plot.line(logx=True)

    plt.xlabel('gamma')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    # experiment_1()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # experiment_6()

    run_experiment({
        'transformer__n_clusters': [760],
        'transformer__n_samples': [100000],
        'transformer__n_levels': [2],
        'transformer__norm': ['power'],
        'classifier__kernel': [histogram_intersection_kernel],
        'classifier__C': [1],
        'classifier__gamma': [.001]
    }, plot_confusion=True)
