import argparse
import os
from shutil import copyfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    parser.add_argument('--train-size', type=int, default=400)
    parser.add_argument('--classes', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    images_per_class = args.train_size // args.classes

    for label in os.listdir(os.path.join(args.src, 'train')):
        os.makedirs(os.path.join(args.dst, 'train', label), mode=744)
        os.makedirs(os.path.join(args.dst, 'test', label), mode=744)

        images = os.listdir(os.path.join(args.src, 'train', label))
        np.random.shuffle(images)

        images_selected = images[:images_per_class]
        images_not_selected = images[images_per_class:]

        for im in images_selected:
            copyfile(os.path.join(args.src, 'train', label, im), os.path.join(args.dst, 'train', label, im))
        for im in images_not_selected:
            copyfile(os.path.join(args.src, 'train', label, im), os.path.join(args.dst, 'test', label, im))

    for label in os.listdir(os.path.join(args.src, 'test')):
        for im in os.listdir(os.path.join(args.src, 'test', label)):
            copyfile(os.path.join(args.src, 'test', label, im), os.path.join(args.dst, 'test', label, im))
