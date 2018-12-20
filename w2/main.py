import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from descriptors.dense_sift import DenseSIFT
from descriptors.spatial_pyramid import SpatialPyramid
from w2.classifier import Classifier
from w2.utils.load_data import load_dataset
from w2.utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    return parser.parse_args()


def main(args):
    parameters = {
        #'classifier__n_neighbors': np.linspace(5, 100, 20)
    }

    pipeline = Pipeline(steps=[
        ("cluster", SpatialPyramid()),
        #("scaler", StandardScaler()),
        ("classifier", Classifier())
    ])

    cv = GridSearchCV(pipeline, parameters, cv=3, verbose=2)

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    sift = DenseSIFT(128)
    with Timer('Extracting fit train descriptors'):
        train_descriptors = sift.compute(train_filenames)
    with Timer('Extracting fit test descriptors'):
        test_descriptors = sift.compute(test_filenames)

    cv.fit(train_descriptors, train_labels)

    # c = Classifier()
    # c.fit(train_filenames, train_labels)

    # TODO print scores
    print(cv.grid_scores_)

    print(cv.score(test_descriptors, test_labels))


if __name__ == '__main__':
    print(main(parse_args()))
