import argparse
import os
import pickle
from typing import Dict, Any

import numpy as np
from keras.utils import plot_model

from model import create_model
from utils import args_to_str, generate_image_patches_db, get_train_generator, get_validation_generator
from utils.metrics import save_confusion_matrix, save_accuracy, save_loss
from tensorflow.python.client import device_lib


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-s', '--image_size', type=int, default=64)
    parser.add_argument('-u', '--units', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('-a', '--activation', type=str, nargs='+', default=['relu', 'relu'])
    parser.add_argument('-l', '--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('-op', '--optimizer', type=str, default='sgd')
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['accuracy'])
    parser.add_argument('-p', '--patches', action='store_true', default=False)
    parser.add_argument('-ps', '--patch_size', type=int, default=64)
    parser.add_argument('-pd', '--patches_dir', type=str, default='/home/grupo06/work/data/MIT_split_patches')
    return parser


def train(args: argparse.Namespace):
    print(device_lib.list_local_devices())
    model = create_model(args.image_size, args.units, args.activation, args.optimizer, args.loss, args.metrics)
    model.summary()

    plot_model(model,
               to_file=os.path.join(args.output_dir, 'modelMLP_{}.png'.format(args_to_str(args))),
               show_shapes=True,
               show_layer_names=True)

    print('Start training...')

    if args.patches:
        if not os.path.exists(args.patches_dir):
            generate_image_patches_db(args.dataset_dir, args.patches_dir, args.patch_size)
        directory = args.patches_dir
        image_size = args.patch_size
    else:
        directory = args.dataset_dir
        image_size = args.image_size
    train_generator = get_train_generator(directory, image_size, args.batch_size)
    validation_generator = get_validation_generator(directory, image_size, args.batch_size)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4
    )

    print('Optimization done!')

    y_pred = model.predict_generator(validation_generator, 807 // args.batch_size + 1)
    y_pred = np.argmax(y_pred, axis=1)
    save_confusion_matrix(validation_generator.classes, y_pred,
                          os.path.join(args.output_dir, 'cm_{}.png'.format(args_to_str(args)))
                          )

    save_accuracy(history, os.path.join(args.output_dir, 'acc_{}.png'.format(args_to_str(args))))
    save_loss(history, os.path.join(args.output_dir, 'loss_{}.png'.format(args_to_str(args))))

    with open(os.path.join(args.output_dir, 'history_{}.pkl'.format(args_to_str(args))), 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)

    model_file = os.path.join(args.output_dir, 'model_{}.h5'.format(args_to_str(args)))
    print('Saving model to {}...'.format(model_file))
    model.save(model_file)

    weights_file = os.path.join(args.output_dir, 'model_{}_weights.h5'.format(args_to_str(args)))
    print('Saving weights to {}...'.format(weights_file))
    model.save_weights(weights_file)

    print('Finished')


def print_setup(args: argparse.Namespace):
    print('\n\n\tExperimental setup')
    print('\t------------------\n')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    print('\n\n')


def main(default_args: Dict[str, Any] = None):
    parser = get_parser()

    # Substitute given arguments if exist
    if default_args:
        args = parser.parse_known_args('')[0]
        a = vars(args)
        a.update(default_args)
        args = argparse.Namespace(**a)
    else:
        args = parser.parse_args()

    print_setup(args)
    train(args)


if __name__ == '__main__':
    main()
