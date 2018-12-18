import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV

from w2.classifier import Classifier
from w2.utils.load_data import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/MIT_split/train')
    parser.add_argument('--test_path', type=str, default='../data/MIT_split/test')
    return parser.parse_args()


def main(args):
    d = {
        'n_neighbors': np.linspace(5, 100, 20)
    }

    cv = GridSearchCV(Classifier(), d, cv=5, n_jobs=4, verbose=2)

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.train_path)
    test_filenames, test_labels = load_dataset(args.test_path)

    #cv.fit(train_filenames, train_labels)

    c = Classifier()
    c.fit(train_filenames, train_labels)

    # TODO print scores
    print(c.score(test_filenames, test_labels))
    # scores = cv.grid_scores_

    cv.score(test_filenames, test_labels)


if __name__ == '__main__':
    print(main(parse_args()))
