import argparse
import cv2

from utils.load_data import load_dataset
from utils.classifier import classifier
from utils.extract_descriptors import extract_descriptors
from timer import Timer
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--n_features', type=int, nargs='+', default=[300])
    parser.add_argument('--n_clusters', type=int, nargs='+', default=128)
    parser.add_argument('--n_neighbors', type=int, nargs='+', default=5)
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
    accuracy = []
    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    for nf in process_arg(args.n_features):
        # Create a SIFT object detector and descriptor.
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nf)

        # Compute the SIFT descriptors for all the train images.
        with Timer('extract train descriptors'):
            train_descriptors = extract_descriptors(sift, train_filenames)

        # Compute the test descriptors.
        with Timer('extract test descriptors'):
            test_descriptors = extract_descriptors(sift, test_filenames)

        for k in process_arg(args.args.n_clusters):
            for n in process_arg(args.n_neighbors):
                # Compute accuracy of the model.
                accuracy.append((
                    nf, k, n,
                    classifier(train_descriptors, train_labels, test_descriptors, test_labels, k, n, args.distance))
                )

    print('accuracy:', accuracy)


if __name__ == '__main__':
    with Timer('total time'):
        main()
