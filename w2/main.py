import argparse

import numpy as np
import pandas
from functional import seq
from joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from descriptors.dense_sift import DenseSIFT
from descriptors.visual_words import SpatialPyramid
from utils.load_data import load_dataset
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
    sift = DenseSIFT(step_size=16, memory=args.cache_path)
    with Timer('Extract train descriptors'):
        train_pictures = sift.compute(train_filenames)
    with Timer('Extract test descriptors'):
        test_pictures = sift.compute(test_filenames)

    train_data = np.array(seq(train_pictures).map(lambda p: p.to_array()).to_list())
    test_data = np.array(seq(test_pictures).map(lambda p: p.to_array()).to_list())

    # Create processing pipeline and run cross-validation.
    transformer = SpatialPyramid(levels=2)
    scaler = StandardScaler()
    classifier = SVC(C=1, gamma=.002)

    memory = Memory(location=args.cache_path, verbose=1)
    pipeline = Pipeline(memory=memory,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])

    le = LabelEncoder()
    le.fit(train_labels)

    cv = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=5, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, le.transform(train_labels))

    with Timer('Test'):
        accuracy = cv.score(test_data, le.transform(test_labels))
    print('Accuracy: {}'.format(accuracy))

    return pandas.DataFrame.from_dict(cv.cv_results_)


if __name__ == '__main__':
    print(main(_parse_args()))
