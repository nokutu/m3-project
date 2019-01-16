import os
import argparse

from keras.utils import plot_model

from model import create_model
from utils import args_to_str, generate_image_patches_db, get_train_generator, get_validation_generator


def parse_args() -> argparse.Namespace:
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
    parser.add_argument('-p', '--patch', action='store_true', default=False)
    parser.add_argument('-ps', '--patch_size', type=int, default=64)
    parser.add_argument('-pd', '--patches_dir', type=str, default='/home/grupo06/work/data/MIT_split_patches')
    return parser.parse_args()


def train(args):
    model = create_model(args.image_size, args.units, args.activation, args.optimizer, args.loss, args.metrics)
    print(model.summary())

    plot_model(model,
               to_file=os.path.join(args.output_dir, 'modelMLP_{}.png'.format(args_to_str(args))),
               show_shapes=True,
               show_layer_names=True)

    print('Start training...')

    if args.patch:
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
        steps_per_epoch=1881 // args.batch_size,
        epochs=50,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=807 // args.batch_size,
        workers=4
    )

    print('Optimization done!')

    model_file = os.path.join(args.output_dir, 'model_{}.h5'.format(args_to_str(args)))
    if os.path.exists(model_file):
        print('WARNING: model file ' + model_file + ' exists and will be overwritten!')
    print('Saving weights to ' + model_file)
    model.save_weights(model_file)  # always save your weights after training or during training


def print_setup(args):
    print('\n\n\tExperimental setup')
    print('\t------------------\n')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    print('\n\n')


def main():
    args = parse_args()
    print_setup(args)
    train(args)


if __name__ == '__main__':
    main()
