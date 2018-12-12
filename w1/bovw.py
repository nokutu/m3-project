import argparse
from itertools import product

import cv2
import pandas

from utils.load_data import load_dataset
from utils.classifier import classifier
from utils.extract_descriptors import extract_descriptors
from timer import Timer
import numpy as np
from sift import sift_method
from dense_sift import DenseSift


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--method', type=str, nargs='+', default=['sift'], choices=['sift', 'dense_sift'])
    parser.add_argument('--n_features', type=int, nargs='+', default=[300])
    parser.add_argument('--step_size', type=int, nargs='+', default=[16])
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[128])
    parser.add_argument('--n_neighbors', type=int, nargs='+', default=[5])
    parser.add_argument('--distance', type=str, nargs='+', default='euclidean')
    return parser.parse_args()


def process_arg(argument):
    if len(argument) == 1:
        arg_list = argument
    elif len(argument) == 3:
        arg_list = np.linspace(argument[0], argument[1], argument[2], dtype=np.int)
    else:
        raise Exception("Invalid argument")
    return arg_list


def main():
    args = parse_args()
    results = []

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    for method_name, nf, ss in product(args.method, process_arg(args.n_features), process_arg(args.step_size)):
        # Create a SIFT object detector and descriptor.
        sift_instance = cv2.xfeatures2d.SIFT_create(nfeatures=nf)

        if method_name == 'sift':
            method = sift_method
        elif method_name == 'dense_sift':
            method = DenseSift(ss).dense_sift
        else:
            raise Exception('Invalid method')

        # Compute the SIFT descriptors for all the train images.
        with Timer('extract train descriptors'):
            train_descriptors = extract_descriptors(method, sift_instance, train_filenames)

        # Compute the test descriptors.
        with Timer('extract test descriptors'):
            test_descriptors = extract_descriptors(method, sift_instance, test_filenames)

        for k, n in product(process_arg(args.n_clusters), process_arg(args.n_neighbors)):
            # Compute accuracy of the model.
            with Timer('classify'):
                results.append((
                    method_name, nf, ss, k, n,
                    classifier(train_descriptors, train_labels, test_descriptors, test_labels, k, n, args.distance))
                )

    data = pandas.DataFrame(results, columns=["method", "n_features", "step_size", "n_clusters", "n_neighbors",
                                              "accuracy"])
    print(data)


if __name__ == '__main__':
    with Timer('total time'):
        main()
