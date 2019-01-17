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
from descriptors.histogram_intersection_kernel import histogram_intersection_kernel
from utils.metrics import save_confusion_matrix
from utils.timer import Timer


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--cache_path', type=str, default='../.cache')
    return parser.parse_args()


def main(args, param_grid=None, plot_confusion=False):
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
    transformer = SpatialPyramid(n_clusters=760, n_samples=100000, n_levels=2, norm='power')
    scaler = StandardScaler(copy=False)
    classifier = SVC(C=1, kernel=histogram_intersection_kernel, gamma=.001)

    memory = Memory(location=args.cache_path, verbose=1)
    pipeline = Pipeline(memory=memory,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])

    cv = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_data, test_labels)

    print('Best params: {}'.format(cv.best_params_))
    print('Accuracy: {}'.format(accuracy))

    if plot_confusion:
        save_confusion_matrix(le.inverse_transform(test_labels), le.inverse_transform(cv.predict(test_data)))

    return pandas.DataFrame.from_dict(cv.cv_results_)


if __name__ == '__main__':
    print(main(_parse_args()))
