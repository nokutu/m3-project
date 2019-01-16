import argparse
import os

from keras.utils import plot_model

from model import model_creation
from model.generator import train_validation_generator
from utils import args_to_str, generate_image_patches_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output', type=str, default='/home/grupo06/work/')
    parser.add_argument('-u', '--units', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('-a', '--activation', type=str, nargs='+', default=['relu', 'relu'])
    parser.add_argument('-l', '--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('-op', '--optimizer', type=str, default='sgd')
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['accuracy'])
    parser.add_argument('-s', '--image-size', type=int, default=64)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-p', '--patch', action='store_true', default=False)
    parser.add_argument('-ps', '--patch-size', type=int, default=64)
    parser.add_argument('-pd', '--patch-dir', type=str, default='/home/grupo06/work/data/MIT_split_patches')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_file = args.output + 'model_' + args_to_str(args) + '.h5'

    model = model_creation(args.image_size, args.units, args.activation, args.loss, args.optimizer, args.metrics)

    print(model.summary())
    plot_model(model, to_file=args.output + 'modelMLP_' + args_to_str(args) + '.png', show_shapes=True,
               show_layer_names=True)
    print('Done!')

    if os.path.exists(model_file):
        print('WARNING: model file ' + model_file + ' exists and will be overwritten!')

    print('Start training...')

    if args.patch:
        if not os.path.exists(args.patch_dir):
            generate_image_patches_db(args.dataset, args.patch_dir, args.patch_size)

        train_generator, validation_generator = train_validation_generator(args.patch_dir, args.patch_size,
                                                                           args.batch_size)
    else:
        train_generator, validation_generator = train_validation_generator(args.dataset, args.image_size,
                                                                           args.batch_size)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // args.batch_size,
        epochs=50,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=807 // args.batch_size)

    print('Done!')
    print('Saving the model into ' + model_file)
    model.save_weights(model_file)  # always save your weights after training or during training
    print('Done!')
