import os
import argparse
import pandas
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from descriptors.histogram_intersection_kernel import histogram_intersection_kernel
from descriptors.dense_sift import DenseSIFT
from descriptors.visual_words import SpatialPyramid
from model.picture_set import PictureSet
from utils.load_data import load_dataset
from utils.timer import Timer


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--cache_path', type=str, default='../data/descriptors')
    return parser.parse_args()


def _load_or_compute(filenames, cache_path, step_size=16):
    cache_file = os.path.join(cache_path, 'descriptors_step_{}.npy'.format(step_size))
    if os.path.exists(cache_file):
        descriptors = np.load(cache_file)
    else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        sift = DenseSIFT(step_size)
        descriptors = sift.compute(filenames)
        np.save(cache_file, descriptors)
    return descriptors


def main(args, param_grid=None):
    # Read the train and test files.
    if param_grid is None:
        param_grid = {}
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    # Compute the Dense SIFT descriptors for all the train and test images.
    with Timer('Extract train descriptors'):
        train_cache_path = os.path.join(args.cache_path, 'train')
        train_descriptors = PictureSet(_load_or_compute(train_filenames, train_cache_path))
    with Timer('Extract test descriptors'):
        test_cache_path = os.path.join(args.cache_path, 'test')
        test_descriptors = PictureSet(_load_or_compute(test_filenames, test_cache_path))

    # Create processing pipeline and run cross-validation.
    transformer = SpatialPyramid(levels=2)
    scaler = StandardScaler()
    classifier = SVC(C=1, kernel=histogram_intersection_kernel, gamma=.002)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=1)
    pipeline = Pipeline(memory=None,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])

    cv = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, refit=True, verbose=2)

    with Timer('Train'):
        cv.fit(train_descriptors, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_descriptors, test_labels)

    # TODO print scores
    # print(cv.cv_results_)

    print('Accuracy: {}'.format(accuracy))

    rmtree(cachedir)

    return pandas.DataFrame.from_dict(cv.cv_results_)


if __name__ == '__main__':
    main(_parse_args())
