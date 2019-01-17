import argparse
from itertools import product

import numpy as np
import pandas

from utils.load_data import load_dataset
from utils.extract_descriptors import SIFT, DenseSIFT
from utils.classifier import Classifier
from utils.metrics import save_confusion_matrix
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--method', type=str, nargs='+', default=['sift'], choices=['sift', 'dense_sift'])
    parser.add_argument('--n_features', type=int, nargs='+', default=[300])
    parser.add_argument('--step_size', type=int, nargs='+', default=[16])
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[128])
    parser.add_argument('--n_neighbors', type=int, nargs='+', default=[5])
    parser.add_argument('--distance', type=str, nargs='+', default=['euclidean'], choices=[
        'euclidean', 'manhattan', 'chebyshev'
    ])
    parser.add_argument('--confusion_matrix', action='store_true')
    return parser.parse_args()


def process_arg(argument):
    if len(argument) == 1:
        arg_list = argument
    elif len(argument) == 3:
        arg_list = np.linspace(argument[0], argument[1], argument[2], dtype=np.int)
    else:
        raise Exception("Invalid argument")
    return arg_list


def run(args):
    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    results = []
    for m, nf, ss in product(args.method, process_arg(args.n_features), process_arg(args.step_size)):
        print('method: {}, n_features: {}, step_size: {}'.format(m, nf, ss))

        if m == 'sift':
            method = SIFT(n_features=nf)
        elif m == 'dense_sift':
            method = DenseSIFT(step_size=ss)
        else:
            raise Exception('Invalid method')

        # Compute the SIFT descriptors for all the train images.
        with Timer('extract train descriptors'):
            train_descriptors = method.compute(train_filenames)

        # Compute the test descriptors.
        with Timer('extract test descriptors'):
            test_descriptors = method.compute(test_filenames)

        for k, n, d in product(process_arg(args.n_clusters), process_arg(args.n_neighbors), args.distance):
            print('n_clusters: {}, n_neighbors: {}, distance: {}'.format(k, n, d))

            # Train the classifier and compute accuracy of the model.
            classifier = Classifier(k, n, d)
            with Timer('train'):
                classifier.train(train_descriptors, train_labels)
            with Timer('test'):
                accuracy = classifier.test(test_descriptors, test_labels)
            results.append((m, d, nf, ss, k, n, accuracy))

            # Optionally plot confusion matrix.
            if args.confusion_matrix:
                save_confusion_matrix(test_labels, classifier.predict(test_descriptors))

    return pandas.DataFrame(results, columns=["method", "distance", "n_features", "step_size", "n_clusters",
                                              "n_neighbors", "accuracy"])


if __name__ == '__main__':
    with Timer('total time'):
        results = run(parse_args())
        print(results)
