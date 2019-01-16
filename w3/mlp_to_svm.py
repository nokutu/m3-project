import argparse
import os

from keras.utils import plot_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from model import model_creation
from utils import args_to_str
from utils.load_data import load_dataset
from utils.timer import Timer

OUTPUT_DIR = '/home/grupo06/work/'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-u', '--units', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('-a', '--activation', type=str, nargs='+', default=['relu', 'relu'])
    parser.add_argument('-l', '--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['accuracy'])
    parser.add_argument('-s', '--image-size', type=int, default=64)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-n', '--names', type=str, nargs='+')
    parser.add_argument('-p', '--patch', type=bool, action='store_true', default=False)
    parser.add_argument('-ps', '--patch-size', type=int, default=64)
    parser.add_argument('-pd', '--patch-dir', type=str, default='/home/grupo06/work/data/MIT_split_patches')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    model_file = OUTPUT_DIR + 'model_' + args_to_str(args) + '.h5'

    # Read the train and test files.
    train_filenames, train_labels = load_dataset(args.dataset + '/train')
    test_filenames, test_labels = load_dataset(args.dataset + '/test')

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    model = model_creation(args.image_size, args.units, args.activation, args.loss, args.optimizer, args.metrics, True)

    print(model.summary())
    plot_model(model, to_file=OUTPUT_DIR + 'modelMLP_' + args_to_str(args) + '.png', show_shapes=True,
               show_layer_names=True)
    print('Done!\n')

    if os.path.exists(model_file):
        print('WARNING: model file ' + model_file + ' exists and will be overwritten!\n')

    print('Start training...\n')

    param_grid = {
        'kernel': ['rbf', 'linear', 'sigmoid']
    }
    cv = GridSearchCV(SVC(), param_grid, n_jobs=-1, cv=3, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_data, test_labels)
