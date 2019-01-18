import argparse

from PIL import Image
from colored import stylize, fg
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from model import build_model
from utils import load_dataset, Timer, str_to_args

import numpy as np

from utils.metrics import save_confusion_matrix

OUTPUT_DIR = '/home/grupo06/work/'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='/home/mcv/datasets/MIT_split')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    last = '_'.join(args.model_file.split('/')[-1].split('.')[0].split('_')[1:])
    last = last.replace('categorical_crossentropy', 'categorical-crossentropy')
    args2 = str_to_args(last)

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.dataset + '/train')
    test_filenames, test_labels = load_dataset(args.dataset + '/test')

    train_ims = np.empty((len(train_filenames), args2.image_size, args2.image_size, 3))
    test_ims = np.empty((len(test_filenames), args2.image_size, args2.image_size, 3))
    for i, imname in enumerate(train_filenames):
        im = Image.open(imname)
        im = im.resize((args2.image_size, args2.image_size))
        train_ims[i, :, :, :] = np.array(im)
    for i, imname in enumerate(test_filenames):
        im = Image.open(imname)
        im = im.resize((args2.image_size, args2.image_size))
        test_ims[i, :, :, :] = np.array(im)

    model = build_model(args2.image_size, args2.units, args2.activation, args2.optimizer, args2.loss, args2.metrics)

    print(stylize('Done!\n', fg('blue')))
    print(stylize("Loading weights from " + args.model_file + ' ...\n', fg('blue')))
    print('\n')

    model.load_weights(args.model_file)

    print(stylize('Done!\n', fg('blue')))

    print(stylize('Start evaluation ...\n', fg('blue')))

    model.outputs.append(model.layers[-2].output)
    train_data = model.predict(train_ims)[1]
    test_data = model.predict(test_ims)[1]

    le = LabelEncoder()
    se = StandardScaler()

    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    se.fit(train_data)
    train_data = se.transform(train_data)
    test_data = se.transform(test_data)

    print('Start training...\n')

    param_grid = {
        'kernel': ['rbf', 'linear', 'sigmoid'],
        'gamma': np.logspace(-3, 9, 5),
        'C': np.logspace(-3, 9, 5)
    }
    cv = GridSearchCV(SVC(), param_grid, n_jobs=3, cv=5, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_data, test_labels)

    print('Best estimator')
    print(cv.best_estimator_)
    print('Test accuracy: ', accuracy)

