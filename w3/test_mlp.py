import os
import sys
import argparse

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.metrics import accuracy_score

from model import load_model_from_weights
from utils.metrics import save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_file', type=str)
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-p', '--patches', action='store_true', default=False)
    return parser.parse_args()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def test(args):
    print('Loading model from {}...'.format(args.weights_file))
    model = load_model_from_weights(args.weights_file)
    model.summary()

    directory = os.path.join(args.dataset_dir, 'test')
    input_size = model.layers[0].input.shape[1:3]

    print('Start evaluation ...')

    classes = {'coast': 0, 'forest': 1, 'highway': 2, 'inside_city': 3, 'mountain': 4, 'Opencountry': 5, 'street': 6, 'tallbuilding': 7}

    actual = []
    predicted = []

    total = 807
    count = 0

    for class_dir in os.listdir(directory):
        cls = classes[class_dir]
        for image_file in os.listdir(os.path.join(directory, class_dir)):
            img = Image.open(os.path.join(directory, class_dir, image_file))
            if args.patches:
                patches = image.extract_patches_2d(np.array(img), input_size, max_patches=128)
                out = model.predict(patches / 255.)
            else:
                img = img.resize(input_size, resample=Image.BICUBIC)
                out = model.predict(np.expand_dims(np.array(img), 0) / 255.)
            predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
            actual.append(cls)
            predicted.append(predicted_cls)
            count += 1
            sys.stdout.write('\rEvaluated images: {}/{}'.format(count, total))
            sys.stdout.flush()

    accuracy = accuracy_score(actual, predicted)
    print('\nTest Acc. = {}'.format(accuracy))

    args_str = os.path.splitext(os.path.basename(args.weights_file))[0].split('_', 1)[1]
    cm_file = os.path.join(args.output_dir, 'cm_{}.png'.format(args_str))
    save_confusion_matrix(actual, predicted, cm_file)


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()
