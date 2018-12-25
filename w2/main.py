import argparse
from shutil import rmtree
from tempfile import mkdtemp

from joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from descriptors.dense_sift import DenseSIFT
from descriptors.visual_words import SpatialPyramid
from utils.load_data import load_dataset
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    return parser.parse_args()


def main(args):
    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    # Compute the Dense SIFT descriptors for all the train and test images.
    sift = DenseSIFT(step_size=16)
    with Timer('Extract train descriptors'):
        train_descriptors = sift.compute(train_filenames)
    with Timer('Extract test descriptors'):
        test_descriptors = sift.compute(test_filenames)

    # Create processing pipeline and run cross-validation.
    transformer = SpatialPyramid(levels=1)
    scaler = StandardScaler()
    classifier = SVC(C=1, kernel='rbf', gamma=.002)

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    pipeline = Pipeline(memory=None,
                        steps=[('transformer', transformer), ('scaler', scaler), ('classifier', classifier)])
    param_grid = {}
    cv = GridSearchCV(pipeline, param_grid, n_jobs=1, cv=3, refit=True, verbose=2)

    with Timer('train'):
        cv.fit(train_descriptors, train_labels)

    with Timer('test'):
        accuracy = cv.score(test_descriptors, test_labels)

    # TODO print scores
    print(cv.cv_results_)

    print('accuracy: {}'.format(accuracy))

    rmtree(cachedir)


if __name__ == '__main__':
    main(parse_args())
