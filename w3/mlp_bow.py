import os
import argparse

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from keras.models import Model, load_model

from utils import load_dataset
from model.bow import BoWTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_file', type=str, required=True)
    return parser.parse_args()


def get_patches(image_file, patch_size):
    img = Image.open(image_file)
    patches = image.extract_patches_2d(np.array(img), (patch_size, patch_size), max_patches=128)
    return patches


def train(args):
    print('Load model...')
    model = load_model(args.model_file)
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    train_filenames, train_labels = load_dataset(os.path.join(args.dataset_dir, 'train'))
    test_filenames, test_labels = load_dataset(os.path.join(args.dataset_dir, 'test'))

    train_inds = np.random.choice(range(len(train_filenames)), 10)
    train_filenames = [train_filenames[i] for i in train_inds]
    train_labels = [train_labels[i] for i in train_inds]
    test_inds = np.random.choice(range(len(test_filenames)), 10)
    test_filenames = [test_filenames[i] for i in test_inds]
    test_labels = [test_labels[i] for i in test_inds]

    print('Split images into patches...')
    train_images = [get_patches(fn, args.patch_size) for fn in train_filenames]
    test_images = [get_patches(fn, args.patch_size) for fn in test_filenames]

    print('Obtain descriptors from patches...')
    train_descriptors = [model.predict(patches / 255., batch_size=128) for patches in train_images]
    test_descriptors = [model.predict(patches / 255., batch_size=128) for patches in test_images]

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    transformer = BoWTransformer()
    scaler = StandardScaler()
    classifier = SVC()
    pipeline = make_pipeline(transformer, scaler, classifier)

    print('Train BoW model...')
    pipeline.fit(train_descriptors, train_labels)

    print('Test BoW model...')
    accuracy = pipeline.score(test_descriptors, test_labels)
    print(accuracy)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
