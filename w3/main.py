import argparse
import os

from keras.utils import plot_model

from model import model_creation
from utils import args_to_str, str_to_args

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
    parser.add_argument('b', '--batch-size', type=int, default=16)
    parser.add_argument('-n', '--names', type=str, nargs='+')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    model_file = OUTPUT_DIR + 'model_' + args_to_str(args) + '.h5'

    model = model_creation(args.image_size, args.units, args.activation, args.loss, args.optimizer, args.metrics)

    print(model.summary())
    plot_model(model, to_file=OUTPUT_DIR + 'modelMLP_' + args_to_str(args) + '.png', show_shapes=True,
               show_layer_names=True)
    print('Done!\n')

    if os.path.exists(model_file):
        print('WARNING: model file ' + model_file + ' exists and will be overwritten!\n')

    print('Start training...\n')
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolders of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR + '/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR + '/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

    print('Done!\n')
    print('Saving the model into ' + model_file + ' \n')
    model.save_weights(model_file)  # always save your weights after training or during training
    print('Done!\n')
