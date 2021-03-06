import argparse
import os

import numpy as np
from PIL import Image
from keras.models import Model, load_model
from sklearn.feature_extraction import image
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from utils import load_dataset
from model import BoWTransformer, histogram_intersection_kernel
from utils.metrics import save_confusion_matrix
from utils import Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('--num_patches', type=int, default=128)
    return parser.parse_args()


class MLP:
    def __init__(self, model_file, num_patches):
        model = load_model(model_file)
        model = Model(inputs=model.input, outputs=model.layers[-2].output)
        model.summary()
        self.model = model
        self.patch_size = model.layers[0].input.shape[1:3]
        self.num_patches = num_patches

    def compute(self, image_files):
        descriptors = np.empty((len(image_files), self.num_patches, self.model.layers[-1].output_shape[1]))

        for i, image_file in enumerate(image_files):
            patches = self._extract_patches(image_file)
            des = self.model.predict(patches / 255., batch_size=self.num_patches)
            descriptors[i, :, :] = des

        return descriptors

    def _extract_patches(self, image_file):
        img = Image.open(image_file)
        patches = image.extract_patches_2d(np.array(img), self.patch_size, max_patches=self.num_patches)
        return patches


def train(args):
    np.random.seed(42)

    train_filenames, train_labels = load_dataset(os.path.join(args.dataset_dir, 'train'))
    test_filenames, test_labels = load_dataset(os.path.join(args.dataset_dir, 'test'))

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    mlp = MLP(args.model_file, args.num_patches)
    transformer = BoWTransformer(n_clusters=760, n_samples=100000, norm='power')
    scaler = StandardScaler(copy=False)

    param_grid = {
        'C': np.logspace(-3, 15, 5, base=2),
        'kernel': [histogram_intersection_kernel],
        'gamma': np.logspace(-15, 3, 5, base=2)
    }
    cv = GridSearchCV(SVC(), param_grid, n_jobs=-1, cv=3, refit=True, verbose=11)

    with Timer('Extract train descriptors'):
        train_descriptors = mlp.compute(train_filenames)
    with Timer('Compute codebook'):
        transformer.fit(train_descriptors)
    with Timer('Compute train visual words'):
        train_descriptors = transformer.transform(train_descriptors)
    with Timer('Compute scaler params'):
        scaler.fit(train_descriptors)
    with Timer('Scale train descriptors'):
        train_descriptors = scaler.transform(train_descriptors)
    with Timer('Train SVM'):
        cv.fit(train_descriptors, train_labels)

    with Timer('Extract test descriptors'):
        test_descriptors = mlp.compute(test_filenames)
    with Timer('Compute test visual words'):
        test_descriptors = transformer.transform(test_descriptors)
    with Timer('Scale test descriptors'):
        test_descriptors = scaler.transform(test_descriptors)
    with Timer('Test SVM'):
        accuracy = cv.score(test_descriptors, test_labels)

    print('Best params: {}'.format(cv.best_params_))
    print('Test accuracy: {}'.format(accuracy))

    args_str = os.path.splitext(os.path.basename(args.model_file))[0].split('_', 1)[1]
    cm_file = os.path.join(args.output_dir, 'cm_{}_bow.png'.format(args_str))
    save_confusion_matrix(test_labels, cv.predict(test_descriptors), cm_file)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
