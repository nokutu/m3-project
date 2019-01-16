import argparse
import os

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image


def args_to_str(args: argparse.Namespace) -> str:
    out = ''
    out += '-'.join(map(str, args.units))
    out += '_' + '-'.join(args.activation)
    out += '_' + args.loss
    out += '_' + args.optimizer
    out += '_' + '-'.join(args.metrics)
    out += '_' + str(args.image_size)
    out += '_' + str(args.batch_size)
    out += '_' + str(args.patch)
    out += '_' + str(args.patch_size)
    return out


def str_to_args(s: str) -> argparse.Namespace:
    args = argparse.Namespace()
    params = s.split('_')

    args.units = map(int, params[0].split('-'))
    args.activation = params[1].split('-')
    args.loss = params[2]
    args.optimizer = params[3]
    args.metrics = params[4].split('-')
    args.image_size = int(params[5])
    args.batch_size = int(params[6])
    args.patch = bool(params[7])
    args.patch_size = int(params[8])

    return args


def generate_image_patches_db(in_directory, out_directory, patch_size=64):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    total = 2688
    count = 0
    for split_dir in os.listdir(in_directory):
        if not os.path.exists(os.path.join(out_directory, split_dir)):
            os.makedirs(os.path.join(out_directory, split_dir))

        for class_dir in os.listdir(os.path.join(in_directory, split_dir)):
            if not os.path.exists(os.path.join(out_directory, split_dir, class_dir)):
                os.makedirs(os.path.join(out_directory, split_dir, class_dir))

            for imname in os.listdir(os.path.join(in_directory, split_dir, class_dir)):
                count += 1
                print('Processed images: ' + str(count) + ' / ' + str(total), end='\r')
                im = Image.open(os.path.join(in_directory, split_dir, class_dir, imname))
                patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=1.0)
                for i, patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(
                        os.path.join(out_directory, split_dir, class_dir, imname.split(',')[0] + '_' + str(i) + '.jpg'))
    print('\n')
