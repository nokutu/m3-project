import argparse
from shutil import rmtree
from tempfile import mkdtemp

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas

from utils.load_data import load_dataset
from descriptors.dense_sift import DenseSIFT
from descriptors.visual_words import SpatialPyramid
from descriptors.histogram_intersection_kernel import histogram_intersection_kernel
from utils.timer import Timer


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    parser.add_argument('--cache_path', type=str, default='../data/cache')
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
        train_descriptors = sift.compute(train_filenames)
    with Timer('Extract test descriptors'):
        test_descriptors = sift.compute(test_filenames)

    # Create processing pipeline and run cross-validation.
    transformer = SpatialPyramid(levels=2)
    scaler = StandardScaler()
    classifier = SVC(C=1, kernel=histogram_intersection_kernel, gamma=.002)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir)
    pipeline = Pipeline(memory=None,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])

    cv = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, refit=True, verbose=2, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_descriptors, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_descriptors, test_labels)
    print('Accuracy: {}'.format(accuracy))

    # Cleanup
    rmtree(cachedir)

    return pandas.DataFrame.from_dict(cv.cv_results_)


if __name__ == '__main__':
    print(main(_parse_args()))
