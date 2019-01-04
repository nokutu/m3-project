import argparse

import numpy as np
from joblib import Memory
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import pandas

from utils.load_data import load_dataset
from descriptors.dense_sift import DenseSIFT
from descriptors.visual_words import SpatialPyramid
from utils.timer import Timer


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--cache_path', type=str, default='../.cache')
    return parser.parse_args()


def main(args, param_grid=None):
    if param_grid is None:
        param_grid = {}

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    # Compute the Dense SIFT descriptors for all the train and test images.
    sift = DenseSIFT(memory=args.cache_path)
    with Timer('Extract train descriptors'):
        train_pictures = sift.compute(train_filenames)
    with Timer('Extract test descriptors'):
        test_pictures = sift.compute(test_filenames)

    train_data = np.array([p.to_array() for p in train_pictures], copy=False)
    test_data = np.array([p.to_array() for p in test_pictures], copy=False)

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    # Create processing pipeline and run cross-validation.
    transformer = SpatialPyramid()
    scaler = StandardScaler(copy=False)
    classifier = SVC(C=1, kernel='rbf', gamma=.001)

    memory = Memory(location=args.cache_path, verbose=0)
    pipeline = Pipeline(memory=memory,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])

    cv = RandomizedSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_data, test_labels)

    print('Best params: {}'.format(cv.best_params_))
    print('Accuracy: {}'.format(accuracy))

    return pandas.DataFrame.from_dict(cv.cv_results_)


if __name__ == '__main__':
    print(main(_parse_args()))
