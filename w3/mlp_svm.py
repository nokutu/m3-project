import argparse

from PIL import Image
from colored import stylize, fg
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from model import create_model
from utils import load_dataset, Timer, str_to_args

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

    train_ims = []
    test_ims = []
    for imname in train_filenames:
        train_ims.append(Image.open(imname))
    for imname in test_filenames:
        test_ims.append(Image.open(imname))

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    model = create_model(args2.image_size, args2.units, args2.activation, args2.optimizer, args2.loss, args2.metrics)

    print(stylize('Done!\n', fg('blue')))
    print(stylize("Loading weights from " + args.model_file + ' ...\n', fg('blue')))
    print('\n')

    model.load_weights(args.model_file)

    print(stylize('Done!\n', fg('blue')))

    print(stylize('Start evaluation ...\n', fg('blue')))

    train_data = model.predict(train_ims)
    test_data = model.predict(test_ims)

    print('Start training...\n')

    param_grid = {
        'kernel': ['rbf', 'linear', 'sigmoid']
    }
    cv = GridSearchCV(SVC(), param_grid, n_jobs=-1, cv=3, refit=True, verbose=11, return_train_score=True)

    with Timer('Train'):
        cv.fit(train_data, train_labels)

    with Timer('Test'):
        accuracy = cv.score(test_data, test_labels)
